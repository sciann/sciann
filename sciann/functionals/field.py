from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import default_bias_initializer, default_kernel_initializer, default_constant_initializer
from ..utils import prepare_default_activations_and_initializers
from ..utils import default_regularizer
from ..utils import floatx, set_floatx

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import linear


class Field(Dense):
    """ Configures the `Field` class for the model outputs.

    # Arguments
        name: String.
            Assigns a layer name for the output.
        units: Positive integer.
            Dimension of the output of the network.
        activation: Callable.
            A callable object for the activation.
        kernel_initializer: Initializer for the kernel.
            Defaulted to a normal distribution.
        bias_initializer: Initializer for the bias.
            Defaulted to a normal distribution.
        kernel_regularizer: Regularizer for the kernel.
            To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
        bias_regularizer: Regularizer for the bias.
            To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
        trainable: Boolean to activate parameters of the network.
        use_bias: Boolean to add bias to the network.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').

    # Raises

    """
    def __init__(self,
                 name=None,
                 units=1,
                 activation=linear,
                 kernel_initializer=None,
                 bias_initializer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 trainable=True,
                 use_bias=True,
                 dtype=None):
        if not dtype:
            dtype = floatx()
        elif not dtype == floatx():
            set_floatx(dtype)

        assert isinstance(name, str), \
            "Please provide a string for field name. "

        # prepare initializers.
        if isinstance(kernel_initializer, (float, int)):
            kernel_initializer = default_constant_initializer(kernel_initializer)
        if isinstance(bias_initializer, (float, int)):
            bias_initializer = default_constant_initializer(bias_initializer)
        if isinstance(kernel_initializer, type(None)) and isinstance(bias_initializer, type(None)):
            activation, kernel_initializer, bias_initializer = [
                a[0] for a in prepare_default_activations_and_initializers(activation)
            ]
        # prepare regularizers.
        kernel_regularizer = default_regularizer(kernel_regularizer)
        bias_regularizer = default_regularizer(bias_regularizer)

        super(Field, self).__init__(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            trainable=trainable,
            use_bias=use_bias,
            name=name,
            dtype=dtype,
        )

    @staticmethod
    def prepare_field_inputs(*args):
        if len(args) == 1 and isinstance(args[0], str):
            fld_name, fld_units = args[0], 1
        elif len(args) == 2 and isinstance(args[0], str):
            fld_name, fld_units = args[0], args[1]
        elif len(args) == 2 and isinstance(args[1], str):
            fld_name, fld_units = args[1], args[0]
        else:
            raise ValueError(
                'Unrecognised inputs for output layer - please use sn.Field to custom define the outputs. '
            )
        return fld_name, fld_units