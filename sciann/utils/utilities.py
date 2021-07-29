""" Built-in utilities to process inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from numpy import pi

import tensorflow as tf
from tensorflow.python import keras as k
from tensorflow.python.keras import backend as K

# interface for some keras features to be acessible across sciann.
from tensorflow.python.keras.backend import is_keras_tensor as is_tensor
from tensorflow.python.keras.backend import floatx
from tensorflow.python.keras.backend import set_floatx
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.initializers import RandomUniform as default_bias_initializer
from tensorflow.python.keras.initializers import GlorotNormal as default_kernel_initializer
from tensorflow.python.keras.initializers import Constant as default_constant_initializer
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.regularizers import l1_l2

from .initializers import SciKernelInitializer as KInitializer
from .initializers import SciBiasInitializer as BInitializer
from .activations import get_activation, SciActivation, SciActivationLayer, SciRowdyActivationLayer
from .validations import is_functional

from pybtex.database.input import bibtex


_DEFAULT_LOG_PATH = ""
_BIBLIOGRAPHY = None
_BIBLIOGRAPHY_TO_OUTPUT = []


def _is_tf_1():
    return tf.__version__.startswith('1.')


def set_random_seed(val=1234):
    """ Set random seed for reproducibility.

    # Arguments
        val: A seed value..

    """
    np.random.seed(val)
    if _is_tf_1():
        tf.set_random_seed(val)
    else:
        tf.random.set_seed(val)


def reset_session():
    """ Clear keras and tensorflow sessions.
    """
    if _is_tf_1():
        K.clear_session()
    else:
        tf.keras.backend.clear_session()


clear_session = reset_session


def set_default_log_path(path):
    global _DEFAULT_LOG_PATH
    _DEFAULT_LOG_PATH = path


def get_default_log_path():
    return _DEFAULT_LOG_PATH


def initialize_bib(bib_file):
    global _BIBLIOGRAPHY
    global _BIBLIOGRAPHY_TO_OUTPUT
    _BIBLIOGRAPHY = bibtex.Parser().parse_file(bib_file)
    _BIBLIOGRAPHY_TO_OUTPUT.append(_BIBLIOGRAPHY.entries['haghighat2021sciann'])
    _BIBLIOGRAPHY_TO_OUTPUT.append(_BIBLIOGRAPHY.entries['raissi2019physics'])


def append_to_bib(bib_entery):
    global _BIBLIOGRAPHY
    global _BIBLIOGRAPHY_TO_OUTPUT
    for bib_entery_i in to_list(bib_entery):
        bib = _BIBLIOGRAPHY.entries[bib_entery_i]
        if bib not in _BIBLIOGRAPHY_TO_OUTPUT:
            _BIBLIOGRAPHY_TO_OUTPUT.append(bib)


def get_bibliography(format='bibtex', file_name=None):
    """Returns the bibliography based on the feastures you used in your model.

    # Argument
        format: 'bibtex', 'bibtextml', 'yaml', ...
            check `pybtex` documentation for other options.
            default: 'bibtex'
        file_name: path to a file.
            default: None. This results in printing the bib file in the outputs.
    """
    global _BIBLIOGRAPHY_TO_OUTPUT
    bib = ""
    for b in _BIBLIOGRAPHY_TO_OUTPUT:
        bib += str(b.to_string(format)) + '\n'

    if file_name is None:
        print(bib)
    else:
        with open(file_name,'w') as f:
            f.write(bib)
            f.close()


def get_log_path(path=None, prefix=None):
    file_path = _DEFAULT_LOG_PATH if path is None else path
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    # add the prefix.
    if prefix is None:
        return file_path
    else:
        return os.path.join(file_path, prefix)


def is_same_tensor(x, y):
    if len(to_list(x)) != len(to_list(y)):
        return False
    else:
        res = []
        for xi, yi in zip(to_list(x), to_list(y)):
            res.append(xi.name == yi.name)
        return all(res)


def unique_tensors(Xs):
    if len(Xs) > 1:
        ux, uids = np.unique([x.name for x in Xs], return_index=True)
        uids = sorted(uids)
        return [Xs[i] for i in uids]
    else:
        return Xs


def default_regularizer(*args, **kwargs):
    l1, l2 = 0.0, 0.0
    if (len(args) == 0 and len(kwargs) == 0) or args[0] is None:
        return None
    elif len(args) == 1:
        if isinstance(args[0], (float, int)):
            l1 = 0.0
            l2 = args[0]
        elif isinstance(args[0], list):
            l1 = args[0][0]
            l2 = args[0][1]
        elif isinstance(args[0], dict):
            l1 = 0.0 if 'l1' not in args[0] else args[0]['l1']
            l2 = 0.0 if 'l2' not in args[0] else args[0]['l2']
    elif len(args) == 2:
        l1 = args[0]
        l2 = args[1]
    elif len(kwargs) > 0:
        l1 = 0.0 if 'l1' not in kwargs else kwargs['l1']
        l2 = 0.0 if 'l2' not in kwargs else kwargs['l2']
    else:
        raise ValueError('Unrecognized entry - input regularization values for l1 and l2.')
    # print("regularization is used with l1={} and l2={}".format(l1, l2))
    return l1_l2(l1=l1, l2=l2)


def default_weight_initializer(actf='linear', distribution='uniform', mode='fan_in', scale=None):
    inz = []
    for i, af in enumerate(to_list(actf)):
        if distribution in ('uniform', 'normal'):
            tp = VarianceScaling(
                scale=eval_default_scale_factor(af, i) if scale is None else scale,
                mode=mode, distribution=distribution
            )
        elif distribution in ('constant',):
            tp = default_constant_initializer(0.0 if scale is None else scale)
        else:
            raise ValueError('Undefined distribution: pick from ("uniform", "normal", "constant").')
        inz.append(tp)
    return inz


def eval_default_scale_factor(actf, lay):
    if actf in ('linear', 'relu'):
        return 2.0
    elif actf in ('tanh', 'sigmoid'):
        return 1.0 if lay > 0 else 1.0
    elif actf in ('sin', 'cos'):
        return 2.0 if lay > 0 else 2.0 #*30.0
    else:
        return 1.0


def prepare_default_activations_and_initializers(actfs, seed=None):
    activations = []
    bias_initializer = []
    kernel_initializer = []
    for lay, actf in enumerate(to_list(actfs)):
        # initializers.
        bias_initializer.append(BInitializer(lay, seed))
        kernel_initializer.append(KInitializer(lay, seed))
        w = kernel_initializer[-1].w0
        # support rowdy net.
        layer_actfs = []
        for sig in to_list(actf):
            if isinstance(sig, str):
                lay_actf = sig.lower().split('-')
                f = get_activation(lay_actf[-1])
            elif callable(sig):
                lay_actf = sig.__name__.lower().split('-')
                f = sig
            else:
                raise TypeError('expected a string for actf: {}'.format(sig))
            if len(lay_actf) == 2:
                append_to_bib("jagtap2020locally")
                layer_actfs.append(SciActivationLayer(w, f, lay_actf[0]))
            else:
                layer_actfs.append(SciActivation(w, f))
        if len(layer_actfs) == 1:
            sig_layer = layer_actfs[0]
        else:
            append_to_bib("jagtap2021deep")
            sig_layer = SciRowdyActivationLayer(layer_actfs)
        activations.append(sig_layer)

    return activations, bias_initializer, kernel_initializer


def unpack_singleton(x):
    """Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument
        x: A list or tuple.

    # Returns
        The same iterable or the first element.
    """
    if len(x) == 1:
        return x[0]
    return x


def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]


def rename(f, name):
    """rename functional.

    # Arguments
        f: A functional object.
        name: A unique (unused) string.

    # Returns
        updated f.
    """
    assert is_functional(f)
    assert str(name)
    f.layers[-1]._name = name
    return f
