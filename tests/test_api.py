import pytest
import sciann as sn
import json
import os
import shutil
from tensorflow.keras import optimizers as tf_optimizers
import numpy as np 


@pytest.fixture(scope="module")
def variable_x():
    return sn.Variable('x')


@pytest.fixture(scope="module")
def variable_y():
    return sn.Variable('y')


@pytest.fixture(scope="module")
def functional_fx(variable_x):
    return sn.Functional('fx', variable_x, 2*[10], 'tanh')


@pytest.fixture(scope="module")
def functional_gx(variable_x):
    return sn.Functional('gx', variable_x, 2*[10], 'tanh')


@pytest.fixture(scope="module")
def functional_fxy(variable_x, variable_y):
    return sn.Functional('fxy', [variable_x, variable_y], 2*[10], 'tanh')


@pytest.fixture(scope="module")
def functional_gxy(variable_x, variable_y):
    return sn.Functional('gxy', [variable_x, variable_y], 2*[10], 'tanh')


@pytest.fixture(scope="module")
def functional_hxy(variable_x, variable_y):
    return sn.Functional('hxy', [variable_x, variable_y], 2*[10], 'tanh', 
                         kernel_initializer=1., bias_initializer=0.)


@pytest.fixture(scope="module")
def functional_hxy_diffs(variable_x, variable_y, functional_hxy):
    h_x = sn.diff(functional_hxy, variable_x)
    h_y = sn.diff(functional_hxy, variable_y)
    h_xx = sn.diff(h_x, variable_x)
    h_yy = sn.diff(h_y, variable_y)
    h_xy = sn.diff(h_x, variable_y)
    return h_x, h_y, h_xx, h_yy, h_xy


@pytest.fixture(scope="module")
def model_fx(variable_x, functional_fx):
    return sn.SciModel(variable_x, functional_fx)


@pytest.fixture(scope="module")
def model_fx_gx(variable_x, functional_fx, functional_gx):
    return sn.SciModel(variable_x, [functional_fx, functional_gx])


@pytest.fixture(scope="module")
def train_data_fx():
    x = np.linspace(-1, 1, 101)
    y = np.tanh(x)
    return x, y


@pytest.fixture(scope="module")
def train_data_fx_gx():
    x = np.linspace(-1, 1, 101)
    y1 = np.tanh(x)
    y2 = np.tanh(x)
    return x, [y1, y2]


@pytest.fixture(scope="module")
def test_data_x():
    x = np.linspace(-1, 1, 101)
    return x


@pytest.fixture(scope="module")
def test_data_xy():
    x = np.linspace(-1, 1, 11)
    y = np.linspace(0, 2, 11)
    return list(np.meshgrid(x, y))


@pytest.fixture(scope="module")
def test_data_xy_dict(test_data_xy):
    return {'x': test_data_xy[0], 'y': test_data_xy[1]}


@pytest.fixture(scope="module")
def test_data_xy_dict_exception(test_data_xy):
    return {'xrand': test_data_xy[0], 'y': test_data_xy[1]}


@pytest.fixture(scope="module")
def expected_hxy(test_data_xy):
    return 10*np.tanh(10*np.tanh(test_data_xy[0] + test_data_xy[1]))


@pytest.fixture(scope="module")
def expected_diff_hxy(test_data_xy):
    x, y = test_data_xy
    h_x = 10*(1 - np.tanh(10*np.tanh(x + y))**2)*(10 - 10*np.tanh(x + y)**2)
    h_y = 10*(1 - np.tanh(10*np.tanh(x + y))**2)*(10 - 10*np.tanh(x + y)**2)
    h_xx = -20*(1 - np.tanh(10*np.tanh(x + y))**2)*(10 - 10*np.tanh(x + y)**2)**2*np.tanh(10*np.tanh(x + y)) - 10*(2 - 2*np.tanh(x + y)**2)*(10 - 10*np.tanh(10*np.tanh(x + y))**2)*np.tanh(x + y)
    h_yy = -20*(1 - np.tanh(10*np.tanh(x + y))**2)*(10 - 10*np.tanh(x + y)**2)**2*np.tanh(10*np.tanh(x + y)) - 10*(2 - 2*np.tanh(x + y)**2)*(10 - 10*np.tanh(10*np.tanh(x + y))**2)*np.tanh(x + y)
    h_xy = -20*(1 - np.tanh(10*np.tanh(x + y))**2)*(10 - 10*np.tanh(x + y)**2)**2*np.tanh(10*np.tanh(x + y)) - 10*(2 - 2*np.tanh(x + y)**2)*(10 - 10*np.tanh(10*np.tanh(x + y))**2)*np.tanh(x + y)
    return [h_x, h_y, h_xx, h_xy, h_yy]


def test_variable():
    xt = sn.Variable('xt', 10, dtype='float32')
    assert sn.is_variable(xt)
    assert sn.is_functional(xt)


def test_variable_exception():
    with pytest.raises(TypeError):
        x = sn.Variable(10)


def test_functional(variable_x):
    x = variable_x
    ft = sn.Functional('ft', x, 2*[10], 'tanh')
    assert sn.is_functional(ft)


def test_functional_exceptions(variable_x):
    x = variable_x
    with pytest.raises(TypeError):
        f = sn.Functional(x)
    with pytest.raises(TypeError):
        ft = sn.Functional('ft', 2*[10])
    with pytest.raises(TypeError):
        ft = sn.Functional('ft', x, 'tanh')
    with pytest.raises(TypeError):
        ft = sn.Functional('ft', x, 2*[10], 12)


def test_variable_operators(variable_x, variable_y):
    assert sn.is_functional(variable_x**2)
    assert sn.is_functional(sn.tanh(variable_x))
    assert sn.is_functional(variable_x + variable_y)
    assert sn.is_functional(sn.tanh(variable_x) + variable_y**2)
    assert sn.is_functional(variable_x + sn.sin(variable_y))
    assert sn.is_functional(2*variable_x + 5)
    assert sn.is_functional(variable_x*2 + 5)


def test_variable_operator_exceptions(variable_x, variable_y):
    with pytest.raises(AttributeError):
        a = variable_x + "to_fail"
    with pytest.raises(AttributeError):
        b = variable_y + None
    with pytest.raises(TypeError):
        c = variable_y**None
    with pytest.raises(TypeError):
        d = variable_y**[1,2,3]


def test_functional_operations(functional_fx, functional_gx, functional_fxy):
    assert sn.is_functional(functional_fx + functional_gx)
    assert sn.is_functional(functional_gx + functional_fxy)
    assert sn.is_functional(functional_fx * functional_gx)
    assert sn.is_functional(functional_fx + functional_gx*functional_fxy)
    assert sn.is_functional(functional_fx / functional_gx)


def test_functional_operation_exceptions(functional_fx, functional_gx, functional_fxy):
    with pytest.raises(TypeError):
        a = functional_fx ** functional_gx


def test_diff(variable_x, variable_y, functional_hxy):
    hxy_x = sn.diff(functional_hxy, variable_x)
    hxy_y = sn.diff(functional_hxy, variable_x)


def test_scimodel(variable_x, variable_y, functional_fx, functional_gx):
    xs = [variable_x, variable_y]
    ys = [functional_fx, functional_gx]
    assert isinstance(sn.SciModel(xs, ys), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "mse", "adam"), sn.SciModel)
    assert isinstance(sn.SciModel(variable_x, ys), sn.SciModel)
    assert isinstance(sn.SciModel(xs, functional_fx), sn.SciModel)


def test_scimodel_exceptions(variable_x, variable_y, functional_fx, functional_gx):
    with pytest.raises(ValueError):
        sn.SciModel(variable_x)
    with pytest.raises(ValueError):
        sn.SciModel(functional_fx)


def test_scimodel_optimizers(variable_x, variable_y, functional_fx, functional_gx):
    xs = [variable_x, variable_y]
    ys = [functional_fx, functional_gx]
    assert isinstance(sn.SciModel(xs, ys, "mse", "adam"), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "mse", "rmsprop"), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "mse", "scipy-l-bfgs-b"), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "mse", "scipy-newtoncg"), sn.SciModel)


def test_scimodel_optimizer_exceptions(variable_x, variable_y, functional_fx, functional_gx):
    xs = [variable_x, variable_y]
    ys = [functional_fx, functional_gx]
    with pytest.raises(ValueError):
        assert isinstance(sn.SciModel(xs, ys, "mse", "to_fail"), sn.SciModel)


def test_scimodel_keras_optimizers(variable_x, variable_y, functional_fx, functional_gx):
    xs = [variable_x, variable_y]
    ys = [functional_fx, functional_gx]
    assert isinstance(sn.SciModel(xs, ys, "mse", tf_optimizers.Adam()), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "mse", tf_optimizers.RMSprop()), sn.SciModel)


def test_scimodel_lossfuncs(variable_x, variable_y, functional_fx, functional_gx):
    xs = [variable_x, variable_y]
    ys = [functional_fx, functional_gx]
    assert isinstance(sn.SciModel(xs, ys, "mse", "adam"), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "mae", "adam"), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "ae", "adam"), sn.SciModel)
    assert isinstance(sn.SciModel(xs, ys, "sse", "adam"), sn.SciModel)


def test_scimodel_lossfunc_exceptions(variable_x, variable_y, functional_fx, functional_gx):
    xs = [variable_x, variable_y]
    ys = [functional_fx, functional_gx]
    with pytest.raises(ValueError):
        assert isinstance(sn.SciModel(xs, ys, "to_fail", "adam"), sn.SciModel)


def test_eval_functional(expected_hxy, test_data_xy, functional_hxy):
    hxy_test = functional_hxy.eval(test_data_xy)
    assert np.linalg.norm(hxy_test - expected_hxy) < 1e-6*np.linalg.norm(expected_hxy), \
        'test_eval_functional: failed using list inputs. '


def test_eval_functional_from_dict(expected_hxy, test_data_xy_dict, functional_hxy):
    hxy_test = functional_hxy.eval(test_data_xy_dict)
    assert np.linalg.norm(hxy_test - expected_hxy) < 1e-6 * np.linalg.norm(expected_hxy), \
        'test_eval_functional: failed using dict inputs. '


def test_eval_functional_from_dict_exception(expected_hxy, test_data_xy_dict_exception, functional_hxy):
    with pytest.raises(ValueError):
        functional_hxy.eval(test_data_xy_dict_exception)


def test_eval_diff(expected_diff_hxy, test_data_xy, functional_hxy_diffs):
    diff_test = [f.eval(test_data_xy) for f in functional_hxy_diffs]
    diff_error = [np.linalg.norm(test - true) < 1e-5*np.linalg.norm(true) 
                  for test, true in zip(diff_test, expected_diff_hxy)]
    assert all(diff_error)


def test_eval_scimodel(test_data_xy, expected_hxy, variable_x, variable_y, functional_hxy):
    m = sn.SciModel([variable_x, variable_y], functional_hxy)
    hxy_test = functional_hxy.eval(m, test_data_xy)
    assert np.linalg.norm(hxy_test - expected_hxy) < 1e-6*np.linalg.norm(expected_hxy)


def test_train(train_data_fx, model_fx):
    h = model_fx.train(train_data_fx[0], train_data_fx[1], epochs=10)


def test_train_learning_rate(train_data_fx, model_fx):
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10, 
        learning_rate = 0.001
    )


def test_train_learning_rate_expscheduler(train_data_fx, model_fx):
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10,
        learning_rate = ([0, 100], [0.001, 0.0001])
    )


def test_train_learning_rate_expscheduler(train_data_fx, model_fx):
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10, 
        learning_rate = {"scheduler": "exponentialdecay", 
                         "initial_learning_rate": 0.001, 
                         "final_learning_rate":0.0001,
                         "decay_epochs": 100}
    )


def test_train_learning_rate_sinescheduler(train_data_fx, model_fx):
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10, 
        learning_rate = {"scheduler": "sineexponentialdecay", 
                         "initial_learning_rate": 0.001, 
                         "final_learning_rate":0.0001,
                         "decay_epochs": 100,
                         "sine_freq": 2,
                         "sine_decay_rate": 0.5}
    )


def test_train_save_weights(train_data_fx, model_fx):
    name_prefix = "weights"
    default_path = os.path.join(os.curdir, name_prefix)
    # test 1: best weights 
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10,
        save_weights={"path":default_path, "freq": 10, "best": True}
    )
    files = list(filter(lambda f: f.startswith(name_prefix + "-best"), os.listdir(os.curdir)))
    assert len(files) > 0
    # test 2: save weights with freq
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10,
        save_weights={"path":default_path, "freq": 10, "best": False}
    )
    files = list(filter(lambda f: f.startswith(name_prefix), os.listdir(os.curdir)))
    assert len(files) > 1
    # delete temporary files. 
    for f in files:
        os.remove(os.path.join(os.curdir, f))


def test_train_log_functionals(train_data_fx, test_data_x, model_fx, functional_fx):
    dir_prefix = "logs"
    default_path = os.path.join(os.curdir, dir_prefix)
    # test 1: best weights 
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10,
        log_functionals={"functionals":[functional_fx], "path": default_path, 
                         "inputs":test_data_x, "freq": 5}
    )
    assert os.path.isdir(default_path)
    files = list(filter(lambda f: f.startswith("functional-history"), os.listdir(default_path)))
    assert len(files) > 1
    shutil.rmtree(default_path)


def test_train_log_loss_landscape(train_data_fx, test_data_x, model_fx, functional_fx):
    dir_prefix = "logs"
    default_path = os.path.join(os.curdir, dir_prefix)
    # test 1: best weights 
    h = model_fx.train(
        train_data_fx[0], train_data_fx[1], epochs=10,
        log_loss_landscape={"norm":2, "resolution":3, "path": default_path, "trials": 3}
    )
    assert os.path.isdir(default_path)
    files = list(filter(lambda f: f.startswith("loss-landscape"), os.listdir(default_path)))
    assert len(files) > 1
    shutil.rmtree(default_path)


def test_train_adaptive_weights(train_data_fx_gx, test_data_x, model_fx_gx):
    # test 1
    h = model_fx_gx.train(
        train_data_fx_gx[0], train_data_fx_gx[1], epochs=10,
        adaptive_weights=True
    )


def test_train_adaptive_weights_gradient_pathology(train_data_fx_gx, test_data_x, model_fx_gx):
    h = model_fx_gx.train(
        train_data_fx_gx[0], train_data_fx_gx[1], epochs=10,
        adaptive_weights={"method": "GP", "freq": 5, "alpha": 1.}
    )


def test_train_adaptive_weights_grad_norm(train_data_fx_gx, test_data_x, model_fx_gx):
    h = model_fx_gx.train(
        train_data_fx_gx[0], train_data_fx_gx[1], epochs=10,
        adaptive_weights={"method": "GN", "freq": 5, "alpha": 1.}
    )


def test_train_adaptive_weights_neural_tangent_kernel(train_data_fx_gx, test_data_x, model_fx_gx):
    h = model_fx_gx.train(
        train_data_fx_gx[0], train_data_fx_gx[1], epochs=10,
        adaptive_weights={"method": "NTK", "freq": 5, "alpha": 1.}
    )


if __name__ == '__main__':
    pytest.main(['--verbose'])
