""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
graph_unique_name = K.get_graph().unique_name

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Lambda
# from keras.layers import BatchNormalization
from keras.models import Model
from tensorflow import tensordot, expand_dims

from ..utils import to_list, unpack_singleton, is_same_tensor, unique_tensors
from ..utils import default_weight_initializer
from ..utils import default_regularizer
from ..utils import validations, getitem
from ..utils import floatx, set_floatx
from ..utils import math
from ..utils.activations import SciActivation, get_activation
from ..utils import prepare_default_activations_and_initializers

from .field import Field
from .mlp_functional import MLPFunctional


""" Configures the Functional object (Neural Network).

# Arguments
    fields: String or Field.
        [Sub-]Network outputs.
        It can be of type `String` - Associated fields will be created internally.
        It can be of type `Field` or `Functional`
    variables: Variable.
        [Sub-]Network inputs.
        It can be of type `Variable` or other Functional objects.
    hidden_layers: A list indicating neurons in the hidden layers.
        e.g. [10, 100, 20] is a for hidden layers with 10, 100, 20, respectively.
    activation: defaulted to "tanh".
        Activation function for the hidden layers.
        Last layer will have a linear output.
    output_activation: defaulted to "linear".
        Activation function to be applied to the network output.
    res_net: (True, False). Constructs a resnet architecture.
        Defaulted to False.
    kernel_initializer: Initializer of the `Kernel`, from `k.initializers`.
    bias_initializer: Initializer of the `Bias`, from `k.initializers`.
    kernel_regularizer: Regularizer for the kernel.
        To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
    bias_regularizer: Regularizer for the bias.
        To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
    trainable: Boolean.
        False if network is not trainable, True otherwise.
        Default value is True.

# Raises
    ValueError:
    TypeError:
"""
def Functional(
        fields=None,
        variables=None,
        hidden_layers=None,
        activation="tanh",
        output_activation="linear",
        res_net=False,
        kernel_initializer=None,
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        trainable=True,
        **kwargs):
    # prepare hidden layers.
    if hidden_layers is None:
        hidden_layers = []
    else:
        hidden_layers = to_list(hidden_layers)
    if not all([isinstance(n, int) for n in hidden_layers]):
        raise TypeError("Enter a list of integers as the third input assigning layer widths, e.g. [10,10,10]. ")
    # prepare kernel initializers.
    activations, def_biasinit, def_kerinit = \
        prepare_default_activations_and_initializers(
        len(hidden_layers) * [activation] + [output_activation]
    )
    if kernel_initializer is None:
        kernel_initializer = def_kerinit
    elif isinstance(kernel_initializer, (float, int)):
        kernel_initializer = default_weight_initializer(
            len(hidden_layers) * [activation] + [output_activation],
            'constant',
            scale=kernel_initializer
        )
    else:
        kernel_initializer = [kernel_initializer for l in len(hidden_layers) * [activation] + [output_activation]]
    # prepare bias initializers.
    if bias_initializer is None:
        bias_initializer = def_biasinit
    elif isinstance(bias_initializer, (float, int)):
        bias_initializer = default_weight_initializer(
            len(hidden_layers) * [activation] + [output_activation],
            'constant',
            scale=bias_initializer
        )
    else:
        bias_initializer = [bias_initializer for l in len(hidden_layers) * [activation] + [output_activation]]
    # prepare regularizers.
    kernel_regularizer = default_regularizer(kernel_regularizer)
    bias_regularizer = default_regularizer(bias_regularizer)
    # prepares fields.
    output_fields = []
    for fld in to_list(fields):
        if isinstance(fld, (str, tuple, list)):
            if isinstance(fld, str):
                fld_name, fld_units = Field.prepare_field_inputs(fld)
            else:
                fld_name, fld_units = Field.prepare_field_inputs(*fld)
            output_fields.append(
                Field(
                    name=fld_name,
                    units=fld_units,
                    kernel_initializer=kernel_initializer[-1],
                    bias_initializer=bias_initializer[-1],
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    trainable=trainable,
                )
            )
        elif validations.is_field(fld):
            output_fields.append(fld)
        else:
            raise TypeError(
                'Please provide a "list" of field names of'
                + ' type "String" or "Field" objects.'
            )
    # prepare inputs/outputs/layers.
    inputs = []
    layers = []
    variables = to_list(variables)
    if all([isinstance(var, MLPFunctional) for var in variables]):
        for var in variables:
            inputs += var.outputs
        # for var in variables:
        #     for lay in var.layers:
        #         layers.append(lay)
    else:
        raise TypeError(
            "Input error: Please provide a `list` of `sn.Variable`s. \n"
            "Provided - {}".format(variables)
        )

    # Input layers.
    if len(inputs) == 1:
        net_input = inputs[0]
    else:
        layer = Concatenate(name=graph_unique_name('concat'))
        net_input = layer(inputs)

    # Define the output network.
    net = [net_input]

    # define the ResNet networks.
    if res_net is True:
        res_layers = []
        res_outputs = []
        for rl in ["U", "V", "H"]:
            layers.append(
                Dense(
                    hidden_layers[0],
                    kernel_initializer=kernel_initializer[0],
                    bias_initializer=bias_initializer[0],
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    trainable=trainable,
                    name=graph_unique_name("DRes"+rl+"{:d}b".format(hidden_layers[0])),
                    **kwargs
                )
            )
            res_output = layers[-1](net_input)
            # Apply the activation.
            if activations[0].activation_name != 'linear':
                layers.append(activations[0])
                res_outputs.append(layers[-1](res_output))
        net[-1] = res_outputs[-1]

    for nLay, nNeuron in enumerate(hidden_layers):
        # Add the layer.
        layer = Dense(
            nNeuron,
            kernel_initializer=kernel_initializer[nLay],
            bias_initializer=bias_initializer[nLay],
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable,
            name=graph_unique_name("D{:d}b".format(nNeuron)),
            **kwargs
        )
        layers.append(layer)
        net[-1] = layer(net[-1])
        # # Add batch normalization. 
        # layer = BatchNormalization()
        # layers.append(layer)
        # net[-1] = layer(net[-1])
        # Apply the activation.
        if activations[nLay].activation_name != 'linear':
            layer = activations[nLay]
            layers.append(layer)
            net[-1] = layer(net[-1])
        # Add the resnet layer
        if res_net is True:
            layer = Lambda(lambda xs: (1-xs[0])*xs[1] + xs[0]*xs[2], name=graph_unique_name("ResLayer"))
            net[-1] = layer([net[-1]] + res_outputs[:2])

    # Assign to the output variable
    if len(net) == 1:
        net_output = net[0]
    else:
        raise ValueError("Legacy for Enrichment: Must be updated. ")
        layer = Concatenate(name=graph_unique_name('concat'))
        net_output = layer(net)

    # Define the final outputs of each network
    functionals = []
    for out in output_fields:
        last_layers = [out]
        last_output = out(net_output)
        # add the activation on the output.
        if activations[-1].activation_name != 'linear':
            layer = activations[-1]
            last_layers.append(layer)
            last_output = layer(last_output)
        # Construct functionals
        functionals.append(
            MLPFunctional(inputs, [last_output], layers + last_layers)
        )

    return unpack_singleton(functionals)

