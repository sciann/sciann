""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
graph_unique_name = K.get_graph().unique_name

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.models import Model
from tensorflow import tensordot, expand_dims
import numpy as np

from ..utils import to_list, unpack_singleton, is_same_tensor, unique_tensors
from ..utils import default_weight_initializer
from ..utils import default_regularizer
from ..utils import validations, getitem
from ..utils import floatx, set_floatx
from ..utils import math
# from ..utils.transformers import FourierFeature
from ..utils.activations import SciActivation, get_activation
from ..utils import prepare_default_activations_and_initializers

from .field import Field


class MLPFunctional(object):
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
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').
            Note: Only network inputs should be set.
        trainable: Boolean.
            False if network is not trainable, True otherwise.
            Default value is True.

    # Raises
        ValueError:
        TypeError:
    """
    def __init__(self, inputs, outputs, layers):
        # check data-type.
        self.inputs = inputs.copy()
        self.outputs = outputs.copy()
        self.layers = layers.copy()
        self._set_model()

    def _set_model(self, inputs, outputs, layers):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers

    def eval(self, *args):
        """ Evaluates the functional object for a given input.

        # Arguments
            (SciModel, Xs): 
                Evalutes the functional object from the beginning 
                    of the graph defined with SciModel. 
                    The Xs should match those of SciModel. 
            
            (Xs):
                Evaluates the functional object from inputs of the functional. 
                    Xs should match those of inputs to the functional.

            (Xs):
                A dictionary, containing values for each variable.
                    
        # Returns
            Numpy array of dimensions of network outputs. 

        # Raises
            ValueError:
            TypeError:
        """
        if len(args) == 1:
            model = self.model
            # read data.
            xs = args[0]
        elif len(args) == 2:
            if validations.is_scimodel(args[0]):
                model = K.function(args[0].model.inputs, self.outputs)
            else:
                raise ValueError(
                    'Expected a SciModel object for the first arg. '
                )
            xs = args[1]
        else:
            raise NotImplemented()

        # To have unified output for postprocessing - limitted support.
        if isinstance(xs, dict):
            xs_names = []
            xs_vals = []
            for x_in in model.inputs:
                x_names = [x_in.name, x_in.name.split(':')[0]]
                if x_names[0] in xs.keys():
                    xs_vals.append(xs[x_names[0]])
                    xs_names.append(x_names[0])
                elif x_names[1] in xs.keys():
                    xs_vals.append(xs[x_names[1]])
                    xs_names.append(x_names[1])
                else:
                    raise ValueError(f'Cannot map network input node {x_names[0]} to the input dict with {xs.keys()}. ')
        elif isinstance(xs, np.ndarray) or isinstance(xs, list):
            xs_vals = to_list(xs)
            assert len(model.inputs) == len(xs_vals), \
                'Number of inputs do not match the number of inputs to the functional. '
        else:
            raise TypeError('Expected a list of `np.ndarray` inputs.')

        shape_default = [x.shape for x in xs_vals]
        assert all([shape_default[0][0] == x[0] for x in shape_default[1:]])

        # prepare X,Y data.
        for i, (x, xt) in enumerate(zip(xs_vals, model.inputs)):
            x_shape = tuple(xt.get_shape().as_list())
            if x.shape[1:] != x_shape[1:]:
                try:
                    xs_vals[i] = xs_vals[i].reshape((-1,) + x_shape[1:])
                except:
                    print(
                        'Could not automatically convert the inputs to be ' 
                        'of the same size as the expected input tensors. ' 
                        'Please provide inputs of the same dimension as the `Variables`. '
                    )
                    assert False

        y_pred = to_list(model(xs_vals))

        # revert back to normal.
        # if isinstance(xs, dict):
        #     for i, (xn, xv) in enumerate(zip(xs_names, xs_vals)):
        #         xs[xn] = xv.reshpae(shape_default[i])
        if isinstance(xs, list):
            for i, x in enumerate(xs):
                xs[i] = x.reshape(shape_default[i])
        
        # return uniform shapes. 
        if all([shape_default[0]==sd for sd in shape_default[1:]]):
            try:
                y_pred = [y.reshape(shape_default[0]) for y in y_pred]
            except:
                print("Input and output dimensions need re-adjustment for post-processing.")

        return unpack_singleton(y_pred)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def model(self):
        self._set_model()
        return self._model

    @property
    def name(self):
        return self._layers[-1].name
    
    def _set_model(self):
        if hasattr(self, '_model'):
            if is_same_tensor(self._inputs, self._model.inputs) and \
               is_same_tensor(self._outputs, self._model.outputs):
               return
        self._model = K.function(
            unique_tensors(self._inputs),
            self._outputs
        )

    def get_weights(self, at_layer=None):
        """ Get the weights and biases of different layers.

        # Arguments
            at_layer: 
                Get the weights of a specific layer. 
            
        # Returns
            List of numpy array. 
        """
        return [l.get_weights() for l in self.layers]

    def set_weights(self, weights):
        """ Set the weights and biases of different layers.

        # Arguments
            weights: Should take the dimensions as the output of ".get_weights"
            
        # Returns 
        """
        try:
            for l, w in zip(self.layers, weights):
                l.set_weights(w)
        except:
            raise ValueError(
                'Provide data exactly the same as .get_weights() outputs. '
            )

    def count_params(self):
        """ Total number of parameters of a functional.

        # Arguments
            
        # Returns 
            Total number of parameters.
        """
        return sum([l.count_params() for l in self.layers])

    def copy(self):
        return MLPFunctional(
            inputs=self.inputs,
            outputs=self.outputs,
            layers=self.layers
        )

    def append_to_layers(self, layers):
        if self.layers is not layers:
            cl = [x.name for x in self.layers]
            for x in layers:
                if not x.name in cl:
                    self.layers += [x]

    def append_to_inputs(self, inputs):
        if self.inputs is not inputs:
            cl = [x.name for x in self.inputs]
            for x in inputs:
                if not x.name in cl:
                    self.inputs.append(x)

    def append_to_outputs(self, outputs):
        self._outputs += to_list(outputs)

    def set_trainable(self, val, layers=None):
        """ Set the weights and biases of a functional object trainable or not-trainable.
        Note: The SciModel should be called after this.

        # Arguments
            val: (Ture, False)
            layers: list of layers to be set trainable or non-trainable.
                defaulted to None.
            
        # Returns 
        """
        print("Warning: Call `model.compile()` after using set_trainable.")
        if isinstance(val, bool):
            if layers is None:
                for l in self._layers:
                    l.trainable = val
            else:
                for li in to_list(layers):
                    i = -1
                    for l in self._layers:
                        if l.count_params() > 0:
                            i += 1
                        if li == i:
                            l.trainable = val
                            break
        else:
            raise ValueError('Expected a boolean value: True or False')

    def reinitialize_weights(self):
        """ Re-initialize the weights and biases of a functional object.

        # Arguments

        # Returns 
        
        """
        for lay in self.layers:
            if hasattr(lay, 'kernel_initializer') and lay.kernel is not None:
                K.set_value(lay.kernel, lay.kernel_initializer(lay.kernel.shape))
            if hasattr(lay, 'bias_initializer') and lay.bias is not None:
                K.set_value(lay.bias, lay.bias_initializer(lay.bias.shape))

    def split(self):
        """ In the case of `Functional` with multiple outputs,
            you can split the outputs and get an associated functional.

        # Returns
            (f1, f2, ...): Tuple of splitted `Functional` objects
                associated to each output.
        """
        if len(self._outputs)==1:
            return self
        fs = ()
        # total number of outputs to get splitted.
        nr = len(self._outputs)
        # associated to each output, there is a layer to be splitted.
        lays = self.layers[:-nr]
        for out, lay in zip(self._outputs, self._layers[-nr:]):
            # copy constructor for functional.
            f = MLPFunctional(
                inputs = to_list(self.inputs),
                outputs = to_list(out),
                layers = lays + to_list(lay)
            )
            fs += (f,)
        return fs

    def __call__(self):
        return self.outputs

    def __pos__(self):
        return self

    def __neg__(self):
        return self*-1.0

    def __add__(self, other):
        return math.add(self, other)

    def __radd__(self, other):
        return math.radd(self, other)

    def __sub__(self, other):
        return math.sub(self, other)

    def __rsub__(self, other):
        return math.rsub(self, other)

    def __mul__(self, other):
        return math.mul(self, other)

    def __rmul__(self, other):
        return math.rmul(self, other)

    def __truediv__(self, other):
        return math.div(self, other)

    def __rtruediv__(self, other):
        return math.rdiv(self, other)

    def __pow__(self, power):
        return math.pow(self, power)

    def __getitem__(self, item):
        return getitem(self, item)

    def diff(self, *args, **kwargs):
        return math.diff(self, *args, **kwargs)

    def __eq__(self, other):
        return math.equal(self, other)

    def __ne__(self, other):
        return math.not_equal(self, other)

    def __gt__(self, other):
        return math.greater(self, other)

    def __ge__(self, other):
        return math.greater_equal(self, other)

    def __lt__(self, other):
        return math.less(self, other)

    def __le__(self, other):
        return math.less_equal(self, other)

    @classmethod
    def get_class(cls):
        return MLPFunctional
