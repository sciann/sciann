from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.layers import InputLayer
from ..utils import floatx, set_floatx, to_list
from .functional import MLPFunctional


class Variable(MLPFunctional):
    """ Configures the `Variable` object for the network's input.

    # Arguments
        name: String.
            Required as derivatives work only with layer names.
        units: Int.
            Number of feature of input var.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').

    # Raises

    """
    def __init__(self,
                 name=None,
                 units=1,
                 dtype=None):

        if not dtype:
            dtype = floatx()
        elif not dtype == floatx():
            set_floatx(dtype)

        layer = InputLayer(
            # batch_input_shape=(None, 1) if units == 1 else (None, units, 1),
            batch_input_shape=(None, units),
            name=name,
            dtype=dtype
        )

        super(Variable, self).__init__(
            layers=to_list(layer),
            inputs=to_list(layer.input),
            outputs=to_list(layer.output)
        )

    @classmethod
    def get_class(cls):
        return MLPFunctional

