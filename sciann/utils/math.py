""" Utilities to process functionals.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
from keras import backend as K
graph_unique_name = K.get_graph().unique_name

from keras.layers import Layer, Lambda, Flatten, Dense
from keras.layers import Dot
from keras.layers import Input
from keras.models import Model
from tensorflow import sin as tf_sin
from tensorflow import cos as tf_cos
from tensorflow import tile as tf_tile
from tensorflow import pow as tf_pow
from tensorflow import divide as tf_divide

from tensorflow import gradients as tf_gradients
from tensorflow import stop_gradient as tf_stop_gradient
from tensorflow import multiply, expand_dims

from .utilities import *
from .validations import *


def fourier(f, w=10, trainable=False):
    """Apply Fourier transform to the `Variable` objects..

    # Arguments
        w: (Int, np.ndarray) Frequencies of transformation.

    # Returns
        A Functional.
    """
    validate_functional(f)
    append_to_bib("wang2020eigenvector")
    layers = []
    outputs = []
    for fi in f.outputs:
        if isinstance(w, int):
            w = 2*np.pi/np.random.rand(w)
        else:
            assert isinstance(w, np.ndarray)
        size_i = fi.shape.as_list()
        assert len(size_i) == 2, 'Designed for the MLP model.'
        assert size_i[-1] == 1, 'Only supported now.'
        layers.append(
            Dense(
                w.size,
                use_bias=False,
                trainable=trainable,
                activation=tf_sin,
                name=graph_unique_name("fourier-sin")
            )
        )
        outputs.append(layers[-1](fi))
        layers[-1].set_weights(
            [w.reshape(layers[-1].weights[0].shape.as_list())]
        )
        layers.append(
            Dense(
                w.size,
                use_bias=False,
                trainable=trainable,
                activation=tf_cos,
                name=graph_unique_name("fourier-cos")
            )
        )
        outputs.append(layers[-1](fi))
        layers[-1].set_weights(
            [w.reshape(layers[-1].weights[0].shape.as_list())]
        )

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(f.inputs.copy()),
        outputs=outputs,
        layers=layers
    )

    return res


def monomial(f, p=10, w=1.):
    """Apply monomial feature transformation to the `Variable` objects..

    # Arguments
        p: (Int, list) monomial powers to be considered.
        w: (Int, list) weights for each term in the monomial(e.g. 1/n! for taylor expansion).

    # Returns
        A Functional.
    """
    validate_functional(f)
    if isinstance(p, int):
        p = list(range(1, p+1))
    else:
        assert isinstance(p, (np.ndarray, list))
    if isinstance(w, float):
        w = len(p)*[w]
    else:
        assert isinstance(w, (np.ndarray, list))
    layers = []
    outputs = []
    for fi in f.outputs:
        f_dim = fi.shape.as_list()
        tile_dim = (len(f_dim)-1)*[1] + [len(p)]
        layers.append(
            Lambda(lambda ys: tf_divide(tf_pow(tf_tile(ys, tile_dim), p), w),
                   name=graph_unique_name("monomials"))
        )
        outputs.append(
            layers[-1](fi)
        )

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(f.inputs.copy()),
        outputs=outputs,
        layers=layers
    )

    return res


def outer(a, b):
    """outer product of two `Functional` objects.

    # Arguments
        a, b: outer(a,b)
        Note that at least one of them should be of type Functional.

    # Returns
        A Functional.
    """
    validate_functional(a)
    validate_functional(b)
    layers = []
    outputs = []
    for a_out in a.outputs:
        for b_out in b.outputs:
            a_shape = a_out.shape.as_list()
            b_shape = b_out.shape.as_list()
            a_exp = len(a_shape)
            b_exp = 1 if b_shape[0] is None else 0
            name = graph_unique_name("outer")
            layers.append(
                Lambda(lambda ys: multiply(expand_dims(ys[0], a_exp),
                                           expand_dims(ys[1], b_exp)), name=name)
            )
            net_output = layers[-1]([a_out, b_out])
            layers.append(Flatten())
            outputs.append(layers[-1](net_output))
    # return the functional
    assert a.get_class() == b.get_class()
    Functional = a.get_class()
    res = Functional(
        inputs=unique_tensors(a.inputs.copy() + b.inputs.copy()),
        outputs=outputs,
        layers=layers
    )
    return res


def pow(a, b):
    """Element-wise exponentiation applied to the `Functional` object.

    # Arguments
        a, b: pow(a,b)
        Note that at least one of them should be of type Functional.

    # Returns
        A Functional.
    """
    if is_functional(a):
        validate_functional(a)
        f, p = a, b
        name = "pow{:d}".format(p) if isinstance(p, int) else "pow{:.3f}".format(p)
        lmbd = [Lambda(lambda x: x ** p, name=graph_unique_name(name)) for X in f.outputs]

    elif is_functional(b):
        validate_functional(b)
        f, p = b, a
        name = "pow{:d}".format(p) if isinstance(p, int) else "pow{:.3f}".format(p)
        lmbd = [Lambda(lambda x: p ** x, name=graph_unique_name(name)) for X in f.outputs]

    else:
        raise ValueError('Expected one functional in the arguments.')

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(f.inputs.copy()),
        outputs = _apply_operation(lmbd, f),
        layers = lmbd
    )

    return res


def add(f, other):
    """Element-wise addition applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: x[0]+x[1], name=graph_unique_name("add")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: x+other, name=graph_unique_name("add")) for X in f.outputs]
    
    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = _apply_operation(lmbd, f, other),
        layers = lmbd
    )
    return res


def radd(f, other):
    """Element-wise right-addition applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    return add(f, other)


def sub(f, other):
    """Element-wise subtraction applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: x[0]-x[1], name=graph_unique_name("sub")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: x-other, name=graph_unique_name("sub")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = _apply_operation(lmbd, f, other),
        layers = lmbd
    )
    return res


def rsub(f, other):
    """Element-wise right-subtraction applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: x[1]-x[0], name=graph_unique_name("rsub")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: other-x, name=graph_unique_name("rsub")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = _apply_operation(lmbd, f, other),
        layers = lmbd
    )
    return res


def mul(f, other):
    """Element-wise multiplication applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: x[0]*x[1], name=graph_unique_name("mul")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: x*other, name=graph_unique_name("mul")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = _apply_operation(lmbd, f, other),
        layers = lmbd
    )
    return res


def rmul(f, other):
    """Element-wise right-multiplication applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    return mul(f, other)


def div(f, other):
    """Element-wise division applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: x[0]/x[1], name=graph_unique_name("div")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: x/other, name=graph_unique_name("div")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = _apply_operation(lmbd, f, other),
        layers = lmbd
    )
    return res


def rdiv(f, other):
    """Element-wise right-division applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: x[1]/x[0], name=graph_unique_name("rdiv")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: other/x, name=graph_unique_name("rdiv")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = _apply_operation(lmbd, f, other),
        layers = lmbd
    )
    return res


def dot(f, other):
    """Dot product of two `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)
    validate_functional(other)
    assert len(f.outputs) == len(other.outputs)

    outputs = []
    layers = []
    for fl, fr in zip(f.outputs, other.outputs):
        assert fl.shape.as_list() == fr.shape.as_list(),\
            'Expected equal dimensions for output of functionals. '
        l = Lambda(
            lambda x: K.reshape(tf.math.reduce_sum(x*fr, list(range(1, len(fl.shape)))), [-1, 1]),
            name=graph_unique_name("dot")
        )
        layers += [l]
        outputs += [l(fl)]
        
    inputs = to_list(f.inputs) + to_list(other.inputs)
    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = outputs,
        layers = layers
    )
    return res


def diag_part(f):
    """Diag_part operation returns diagonal part of outputs of (None,N,N) functional.

    # Arguments
        f: Functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    lmbd = []
    outputs = []
    for o in f.outputs:
        assert len(o.shape) == 3, \
            'Exptected output dimension to be (None, N, N)'
        dim = o.shape[-1]
        l = Lambda(
            lambda x: tf.linalg.diag_part(x),
            name=graph_unique_name("diag_part")
        )
        lmbd += [l]
        outputs += [l(o)]

    Functional = f.get_class()
    res = Functional(
        inputs = f.inputs.copy(),
        outputs = outputs,
        layers = lmbd
    )
    return res


def diag(f):
    """Diag operation converts a vector output (None, N) to a matrix form of (None,N,N) functional.

    # Arguments
        f: Functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    lmbd = []
    outputs = []
    for o in f.outputs:
        assert len(o.shape) == 2, \
            'Exptected output dimension to be (None, N)'
        dim = o.shape[-1]
        l = Lambda(
            lambda x: tf.linalg.diag(x),
            name=graph_unique_name("diag")
        )
        lmbd += [l]
        outputs += [l(o)]

    Functional = f.get_class()
    res = Functional(
        inputs = f.inputs.copy(),
        outputs = outputs,
        layers = lmbd
    )
    return res


def _apply_operation(lambda_layer, lhs, rhs=None):
    """Element-wise mathematical operation applied on the `Functional` objects.

    # Arguments
        lambda_layer: the layers to perform the operation.
        lhs: left hand side objects.
        rhs: right hand side objects.

    # Returns
        output tensors.
    """
    validate_functional(lhs)

    if is_functional(rhs):
        outputs = [l([x, y]) for l, x, y in zip(lambda_layer, lhs.outputs, rhs.outputs)]
    else:
        try:
            outputs = [l(x) for l, x in zip(lambda_layer, lhs.outputs)]
        except (ValueError, TypeError):
            print(
                'Unsupported operation with an object of type {}. '.format(type(lhs))
            )
            outputs = None

    return outputs


def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sin')


def asin(x):
    """Computes asin of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'asin')


def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'cos')


def acos(x):
    """Computes acos of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'acos')


def tan(x):
    """Computes tan of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'tan')


def atan(x):
    """Computes atan of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'atan')


def atan2(y, x):
    """Computes atan2 of y, x pair, element-wise.

    # Arguments
        y: Functional object.
        x: Functional object.

    # Returns
        A new functional object.
    """
    validate_functional(x)
    validate_functional(y)

    fun = get_activation('atan2')
    lmbd, outputs = [], []
    for i in range(len(x.outputs)):
        lmbd.append(
            Lambda(
                lambda xs: fun(xs[0], xs[1]),
                name=graph_unique_name("{}".format('atan2'))
            )
        )
        outputs += [lmbd[-1]([yi, xi]) for yi, xi in zip(y.outputs, x.outputs)]
    Functional = x.get_class()
    inputs = y.inputs.copy() + x.inputs.copy()
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = outputs,
        layers = lmbd
    )
    return res


def cot(x):
    """Computes cot of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'cot')


def acot(x):
    """Computes acot of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'acot')


def sinh(x):
    """Computes sinh of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sinh')


def cosh(x):
    """Computes cosh of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'cosh')


def tanh(x):
    """Computes tanh of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'tanh')


def coth(x):
    """Computes coth of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'coth')


def sigmoid(x):
    """Computes coth of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sigmoid')


def abs(x):
    """Computes abs of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'abs')


def sign(x):
    """Computes sign of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sign')


def step(x):
    """Computes step (Heaviside) of x element-wise.
       H(x) = 0 if x<=0
       H(x) = 1 if x>0

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    validate_functional(x)

    lmbd = []
    for i in range(len(x.outputs)):
        lmbd.append(
            Lambda(
                lambda x: K.cast(K.greater(x, 0.0), x.dtype), 
                name=graph_unique_name("step")
            )
        )
        
    Functional = x.get_class()
    res = Functional(
        inputs = x.inputs.copy(),
        outputs = _apply_operation(lmbd, x),
        layers = lmbd
    )
    return res

def step8(x):
    """Computes step (Heaviside) of x element-wise.
       H(x) = 0 if x<=0
       H(x) = 1 if x>0

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    validate_functional(x)

    lmbd = []
    for i in range(len(x.outputs)):
        lmbd.append(
            Lambda(
                lambda x: K.cast(K.greater(x, 8.0), x.dtype), 
                name=graph_unique_name("step8")
            )
        )
        
    Functional = x.get_class()
    res = Functional(
        inputs = x.inputs.copy(),
        outputs = _apply_operation(lmbd, x),
        layers = lmbd
    )
    return res
def step12(x):
    """Computes step (Heaviside) of x element-wise.
       H(x) = 0 if x<=0
       H(x) = 1 if x>0

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    validate_functional(x)

    lmbd = []
    for i in range(len(x.outputs)):
        lmbd.append(
            Lambda(
                lambda x: K.cast(K.greater(x, 12.0), x.dtype), 
                name=graph_unique_name("step12")
            )
        )
        
    Functional = x.get_class()
    res = Functional(
        inputs = x.inputs.copy(),
        outputs = _apply_operation(lmbd, x),
        layers = lmbd
    )
    return res

def log(x):
    """Computes log of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'log')


def log10(x):
    """Computes log10 of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return log(x)/np.log(10.0)


def exp(x):
    """Computes exp of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'exp')


def sqrt(x):
    """Computes sqrt of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sqrt')


def square(x):
    """Computes square of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'square')


def relu(x):
    """Computes relu of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'relu')


def mean(x, **kwargs):
    """Apply mean to the `Functional` objects on far-right axis.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    if "axis" not in kwargs:
        kwargs["axis"] = -1
    if "keepdims" not in kwargs:
        kwargs["keepdims"] = True
    return _apply_function(x, 'mean', **kwargs)


def equal(f, other, tol=None):
    """Element-wise comparison applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.
        tol: (float) If you need a tolerance measure.

    # Returns
        A Functional.
    """
    validate_functional(f)
    assert isinstance(tol, (type(None), float)), 'Expected a floating value for `tol`.'

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        if tol is None:
            lambda_opr = lambda x: K.cast_to_floatx(K.equal(x[0], x[1]))
        else:
            lambda_opr = lambda x: K.cast_to_floatx(K.less_equal(K.abs(x[0]-x[1]), tol))
    else:
        _warn_for_ndarray(other)
        if tol is None:
            lambda_opr = lambda x: K.cast_to_floatx(K.equal(x, other))
        else:
            lambda_opr = lambda x: K.cast_to_floatx(K.less_equal(K.abs(x-other), tol))

    lmbd = [Lambda(lambda_opr, name=graph_unique_name("equal")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def not_equal(f, other, tol=None):
    """Element-wise comparison applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.
        tol: (float) If you need a tolerance measure.

    # Returns
        A Functional.
    """
    validate_functional(f)
    assert isinstance(tol, (type(None), float)), 'Expected a floating value for `tol`.'

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        if tol is None:
            lambda_opr = lambda x: K.cast_to_floatx(K.not_equal(x[0], x[1]))
        else:
            lambda_opr = lambda x: K.cast_to_floatx(K.greater(K.abs(x[0] - x[1]), tol))
    else:
        _warn_for_ndarray(other)
        if tol is None:
            lambda_opr = lambda x: K.cast_to_floatx(K.not_equal(x, other))
        else:
            lambda_opr = lambda x: K.cast_to_floatx(K.greater(K.abs(x - other), tol))

    lmbd = [Lambda(lambda_opr, name=graph_unique_name("not_equal")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def greater(f, other):
    """Element-wise comparison applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.greater(x[0], x[1])), name=graph_unique_name("greater")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.greater(x, other)), name=graph_unique_name("greater")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


# legacy support
tol_equal = equal


def greater_equal(f, other):
    """Element-wise comparison applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.greater_equal(x[0], x[1])), name=graph_unique_name("greater_equal")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.greater_equal(x, other)), name=graph_unique_name("greater_equal")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def less(f, other):
    """Element-wise comparison applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.less(x[0], x[1])), name=graph_unique_name("less")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.less(x, other)), name=graph_unique_name("less")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def less_equal(f, other):
    """Element-wise comparison applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    inputs = f.inputs.copy()
    if is_functional(other):
        inputs += to_list(other.inputs)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.less_equal(x[0], x[1])), name=graph_unique_name("less_equal")) for X in f.outputs]
    else:
        _warn_for_ndarray(other)
        lmbd = [Lambda(lambda x: K.cast_to_floatx(K.less_equal(x, other)), name=graph_unique_name("less_equal")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def logical_and(f, other):
    """Element-wise logical-and to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)
    validate_functional(other)
    inputs = f.inputs.copy() + to_list(other.inputs)

    lambda_opr = lambda x: K.cast_to_floatx(tf.logical_and(K.cast(x[0], bool), K.cast(x[1], bool)))
    lmbd = [Lambda(lambda_opr, name=graph_unique_name("logical_and")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def logical_or(f, other):
    """Element-wise logical-or to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)
    validate_functional(other)
    inputs = f.inputs.copy() + to_list(other.inputs)

    lambda_opr = lambda x: K.cast_to_floatx(tf.logical_or(K.cast(x[0], bool), K.cast(x[1], bool)))
    lmbd = [Lambda(lambda_opr, name=graph_unique_name("logical_or")) for X in f.outputs]

    Functional = f.get_class()
    res = Functional(
        inputs=unique_tensors(inputs),
        outputs=_apply_operation(lmbd, f, other),
        layers=lmbd
    )
    return res


def _apply_function(x, fname, **kwargs):
    """Apply `fname` function to x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    validate_functional(x)

    fun = get_activation(fname)
    lmbd = []
    for i in range(len(x.outputs)):
        lmbd.append(
            Lambda(
                lambda x: fun(x, **kwargs),
                name=graph_unique_name("{}".format(fname))
            )
        )
    Functional = x.get_class()
    res = Functional(
        inputs = x.inputs.copy(),
        outputs = _apply_operation(lmbd, x),
        layers = lmbd
    )
    return res


def getitem(x, item):
    """returns specific item of a tensor (Functional).

    # Arguments
        item: Item list.

    # Returns
        A new functional object.
    """
    validate_functional(x)

    in_item = item
    print(in_item)
    if not isinstance(in_item, tuple):
        in_item = (in_item,)
    print(in_item)

    itms = (slice(None, None, None),)
    for it in in_item:
        itms += (slice(it, it+1) if isinstance(it, int) else it, )
    
    lmbd = []
    ys = []
    for y in x.outputs:
        l = Lambda(
            lambda xx: xx[itms], 
            name=graph_unique_name("slice")
        )
        lmbd.append(l)
        ys.append(l(y))

    Functional = x.get_class()
    res = Functional(
        inputs = x.inputs.copy(),
        outputs = ys,
        layers = lmbd
    )
    return res


def _gradients(ys, xs, order=1):
    """Returns the gradients of y in `ys` w.r.t. x in `xs`.

    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        order: Order of differentiation.

    # Returns
        A list of `D^n y / Dx^n` for each y and x in `ys` and `xs`.
    """
    if ys.shape[-1] == 1:
        ds = ys
        for i in range(order):
            ds = unpack_singleton(
                tf_gradients(
                    ds, xs,
                    unconnected_gradients='zero',
                    # colocate_gradients_with_ops=True, TF: V1.14.0
                )
            )

    else:
        splitted_ys = tf.split(ys, num_or_size_splits=ys.shape[-1], axis=-1)
        ds = []
        for j, y in enumerate(splitted_ys):
            ds.append(y)
            for i in range(order):
                ds[-1] = unpack_singleton(
                    tf_gradients(
                        ds[-1], xs,
                        unconnected_gradients='zero',
                        # colocate_gradients_with_ops=True, TF: V1.14.0
                    )
                )
            new_shape = [x if x is not None else -1 for x in ds[-1].shape + (1,)]
            ds[-1] = K.reshape(ds[-1], new_shape)
        # The output is a tensor.
        ds = K.concatenate(ds, -1)
    return ds


def _diag_gradients(ys, xs, order=1):
    """Returns the gradients of y in `ys` w.r.t. x in `xs`.

    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        order: Order of differentiation.

    # Returns
        A list of `D^n y / Dx^n` for each y and x in `ys` and `xs`.
    """
    assert ys.shape.as_list() == xs.shape.as_list(), \
        'Supported when X and Y has the same dimensions - ' + \
        'Xs:{}, Ys:{}'.format(xs.shape.as_list(), ys.shape.as_list())

    ds = _gradients(ys, xs, order)
    return tf.linalg.diag_part(ds)


def _diag_gradients2(ys, xs, order=1):
    """Returns the gradients of y in `ys` w.r.t. x in `xs`.

    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        order: Order of differentiation.

    # Returns
        A list of `D^n y / Dx^n` for each y and x in `ys` and `xs`.
    """
    assert ys.shape.as_list() == xs.shape.as_list(), \
        'Supported when X and Y has the same dimensions - ' + \
        'Xs:{}, Ys:{}'.format(xs.shape.as_list(), ys.shape.as_list())

    splitted_ys = tf.split(ys, num_or_size_splits=ys.shape[-1], axis=-1)
    ds = []
    for j, y in enumerate(splitted_ys):
        ds.append(y)
        for i in range(order):
            ds[-1] = unpack_singleton(
                tf.gradients(
                    ds[-1], xs,
                    unconnected_gradients='zero',
                    # colocate_gradients_with_ops=True, TF: V1.14.0
                )
            )
            
        ds[-1] = ds[-1][:, j:j+1]
    # The output is a tensor.
    ds = K.concatenate(ds, -1)
    return ds


def _div_gradients(ys, xs, order=1):
    """Returns the gradients of y in `ys` w.r.t. x in `xs`.

    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        order: Order of differentiation.

    # Returns
        A list of `D^n y / Dx^n` for each y and x in `ys` and `xs`.
    """
    assert ys.shape.as_list() == xs.shape.as_list(), \
        'Supported when X and Y has the same dimensions - ' + \
        'Xs:{}, Ys:{}'.format(xs.shape.as_list(), ys.shape.as_list())

    ds = _diag_gradients(ys, xs, order)
    return tf.math.reduce_sum(ds, [1], keepdims=True)


def _lambda_gradient(ys, xs, order=1, gtype='Grad', name=''):
    """Returns the gradients of y in `ys` w.r.t. x in `xs` using Lambda layers.
    
    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        gtype: type of differentiation - can be:
            - 'Grad' for gradient operation, i.e. Gij = dy_j / dx_i
            - 'dGrad' for the diagonal of gradient tensor, i.e. Gi = dy_i / dx_i
            - 'Div' for divergence operation, i.e. G = sum(dy_i / dx_i)
        name: A str name for the Lambda layer. 

    # Returns
        A tuple, `(layers, grads)`.
        layers: A Lambda layer or list of Lambda layers where the gradient operator is applied.
        grads: A gradient tensor or list of gradient tensors. 
    """
    
    grads, layers = [], []
    for y in to_list(ys):
        if gtype == 'Grad':
            name_prefix = 'Grad_' if order == 1 else 'Grad{:d}_'.format(order) + name + '/'
            lay = Lambda(lambda y: _gradients(y, xs, order), name=graph_unique_name(name_prefix))
        elif gtype == 'dGrad':
            name_prefix = 'DiagGrad_' if order == 1 else 'Grad{:d}_'.format(order) + name + '/'
            lay = Lambda(lambda y: _diag_gradients(y, xs, order), name=graph_unique_name(name_prefix))
        elif gtype == 'Div':
            name_prefix = 'Div_' if order == 1 else 'Grad{:d}_'.format(order) + name + '/'
            lay = Lambda(lambda y: _div_gradients(y, xs, order), name=graph_unique_name(name_prefix))
        else:
            raise ValueError(
                'Unrecognised gradient type: {} \n'.format(type) +
                '     Please choose among (Grad, dGrad, Div). '
            )
        layers += to_list(lay)
        grads += to_list(lay(y))

    return (unpack_singleton(layers), unpack_singleton(grads))


def _gdiff(gtype, f, *args, **kwargs):
    """Computes gradient of functional object f.

    # Arguments
        gtype: gradient type - choose from (Grad, dGrad, Div)
        f: Functional object.
        ys: layer name for `ys` to differentiate.
        xs: layer name for `xs` to be differentiated w.r.t.
        order: order of differentiation w.r.t. xs - defaulted to 1.

    # Returns
        A new functional object.
    """

    assert is_functional(f), \
        'Please provide a proper functional object. '
    assert(len(args) in (1,2)), \
        'Expected (`ys`, `xs`) or `xs` as the input, '\
        'was provided {:d} inputs'.format(len(args))
    if not all([isinstance(v, str) or is_functional(v) for v in args]):
        raise ValueError(
            'Expected a `Layer` name for a `Functional` to perform differentitation.'
        )

    try:
        inputs = f.inputs.copy()
        if len(args) == 0:
            x = unpack_singleton(f.inputs)
            assert is_tensor(x), \
                'multiple inputs detected - please provide an `x` name. '
            x_name = x.name.split('/')[0]
        else:
            x_id = 0 if len(args)==1 else 1
            if isinstance(args[x_id], str):
                x_name = args[x_id]
                x = next(l for l in f.layers if l.name == x_name).output
            elif is_functional(args[x_id]):
                inputs += to_list(args[x_id].inputs)
                x_lay = args[x_id].layers[-1]
                x_name = x_lay.name
                x = x_lay.output
            else:
                raise TypeError('Unsupported `x` entry. ')
            
        if len(args) <= 1:
            y = unpack_singleton(f.outputs)
            assert is_tensor(y), \
                'multiple outputs detected - please provide a `y` name. '
            y_name = y.name.split('/')[0]
        else:
            y_id = 0
            if isinstance(args[y_id], str):
                y_name = args[y_id]
                y = next(l for l in f.layers if l.name == y_name).output
            elif is_functional(args[y_id]):
                y_lay = args[y_id].layers[-1]
                y_name = y_lay.name
                y = y_lay.output
            else:
                raise TypeError('Unsupported `y` entry. ')
        
    except (StopIteration, ValueError):
        print("Did not find the layer {}. ".format(args))

    # check order of differentiation.
    order = 1
    if 'order' in kwargs.keys():
        order = kwargs['order']
    
    lay, tens = _lambda_gradient(
        y, x, order, gtype, "{}_{}".format(y_name, x_name)
    )
    
    Functional = type(f)
    res = Functional(
        inputs = unique_tensors(inputs),
        outputs = to_list(tens),
        layers = to_list(lay)
    )
    
    return res


def grad(f, *args, **kwargs):
    """Computes gradient tensor of functional object f.

    # Arguments
        f: Functional object.
        ys: layer name for `ys` to differentiate.
        xs: layer name for `xs` to be differentiated w.r.t.
        order: order of differentiation w.r.t. xs - defaulted to 1.

    # Returns
        A new functional object.
    """
    return _gdiff("Grad", f, *args, **kwargs)


# overlaod for backward compatibility 
diff = grad


# consistency with older versions.
def diag_grad(f, *args, **kwargs):
    """Computes diag of gradient tensor of functional object f.

    # Arguments
        f: Functional object.
        ys: layer name for `ys` to differentiate.
        xs: layer name for `xs` to be differentiated w.r.t.
        order: order of differentiation w.r.t. xs - defaulted to 1.

    # Returns
        A new functional object.
    """
    return _gdiff("dGrad", f, *args, **kwargs)


# consistency with older versions.
def div_grad(f, *args, **kwargs):
    """Computes Divergence of functional object f.

    # Arguments
        f: Functional object.
        ys: layer name for `ys` to differentiate.
        xs: layer name for `xs` to be differentiated w.r.t.
        order: order of differentiation w.r.t. xs - defaulted to 1.

    # Returns
        A new functional object.
    """
    return _gdiff("Div", f, *args, **kwargs)


def stop_gradient(f):
    """Equivalent to stop_grdient in tf.
    Use-case: to force zero-gradient with respect to a part of the network.

    # Arguments
        f: Functional object.

    # Returns
        A new functional object.
    """
    if is_functional(f):
        lmbd = [Lambda(lambda x: tf_stop_gradient(x), name=graph_unique_name("stop_grad")) for X in f.outputs]
    else:
        raise ValueError('Expected one functional in the arguments.')

    Functional = f.get_class()
    res = Functional(
        inputs = unique_tensors(f.inputs.copy()),
        outputs = _apply_operation(lmbd, f),
        layers = lmbd
    )

    return res


def radial_basis(xs, ci, radii):
    """Apply `radial_basis` function to x element-wise.

    # Arguments
        xs: List of functional objects.
        ci: Center of basis functional (same length as xs).
        radii: standard deviation or radius from the center.

    # Returns
        A new functional object.
    """
    assert len(xs) == len(ci)
    assert radii > 0.0
    assert all([is_variable(x) for x in xs])
    assert isinstance(xs, list) and isinstance(ci, list)

    for x in xs:
        validate_variable(x)

    return exp(-sum([(x - c)**2 for x, c in zip(xs, ci)])/radii**2)


def radial_basis2(xs, ci, radii):
    """Apply `radial_basis` function to x element-wise.

    # Arguments
        xs: List of functional objects.
        ci: Center of basis functional (same length as xs).
        radii: standard deviation or radius from the center.

    # Returns
        A new functional object.
    """
    assert len(xs) == len(ci)
    assert radii > 0.0
    assert all([is_variable(x) for x in xs])
    assert isinstance(xs, list) and isinstance(ci, list)

    for x in xs:
        validate_variable(x)

    d = xs[0] - ci[0]
    for i in range(1, len(xs)):
        d += xs[i] - ci[i]
    d /= radii
    
    return exp(-sum([(x - c)**2 for x, c in zip(xs, ci)])/radii**2)


def _warn_for_ndarray(other):
    if isinstance(other, np.ndarray):
        Warning(
            'Expecting `Tensor` objects instead of `ndarray`: ' +
            'Note data should go to the training process and ' +
            'this operation may break batch training. '
        )

