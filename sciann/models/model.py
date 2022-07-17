""" SciModel class to define and train the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from keras import backend as K
from tensorflow.python import keras as k
import numpy as np

from keras.utils.data_utils import Sequence
from keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow import gradients as tf_gradients

from ..utils import unpack_singleton, to_list
from ..utils import is_variable, is_constraint, is_functional
from ..utils.optimizers import GradientObserver, ScipyOptimizer, GeneratorWrapper
from ..utils.callbacks import *
from ..utils.schedules import setup_lr_scheduler

from ..functionals import Variable
from ..functionals import RadialBasis
from ..constraints import Data, PDE, Tie


class SciModel(object):
    """Configures the model for training.
    Example:
    # Arguments
        inputs: Main variables (also called inputs, or independent variables) of the network, `xs`.
            They all should be of type `Variable`.
        targets: list all targets (also called outputs, or dependent variables)
            to be satisfied during the training. Expected list members are:
            - Entries of type `Constraint`, such as Data, Tie, etc.
            - Entries of type `Functional` can be:
                . A single `Functional`: will be treated as a Data constraint.
                    The object can be just a `Functional` or any derivatives of `Functional`s.
                    An example is a PDE that is supposed to be zero.
                . A tuple of (`Functional`, `Functional`): will be treated as a `Constraint` of type `Tie`.
            - If you need to impose more complex types of constraints or
                to impose a constraint partially in a specific part of region,
                use `Data` or `Tie` classes from `Constraint`.
        loss_func: defaulted to "mse" or "mean_squared_error".
            It can be an string from supported loss functions, i.e. ("mse" or "mae").
            Alternatively, you can create your own loss function and
            pass the function handle (check Keras for more information).
        optimizer: defaulted to "adam" optimizer.
            It can be one of Keras accepted optimizers, e.g. "adam".
            You can also pass more details on the optimizer:
            - `optimizer = k.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)`
            - `optimizer = k.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)`
            - `optimizer = k.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)`
            Check our Keras documentation for further details. We have found
        load_weights_from: (file_path) Instantiate state of the model from a previously saved state.
        plot_to_file: A string file name to output the network architecture.

    # Raises
        ValueError: `inputs` must be of type Variable.
                    `targets` must be of types `Functional`, or (`Functional`, data), or (`Functional`, `Functional`).
    """
    def __init__(self,
                 inputs=None,
                 targets=None,
                 loss_func="mse",
                 optimizer="adam",
                 load_weights_from=None,
                 plot_to_file=None,
                 **kwargs):
        # strictly check for inputs to be of type variable.
        inputs = to_list(inputs)
        if not all([is_variable(x) for x in inputs]):
            raise ValueError(
                'Please provide a `list` of `Variable` or `RadialBasis` objects for inputs. '
            )
        # prepare input tensors.
        input_vars = []
        for var in inputs:
            input_vars += var.inputs
        # check outputs if of correct type.
        if targets is None:
            if 'constraints' in kwargs:
                targets = kwargs.get('constraints')
            elif 'conditions' in kwargs:
                targets = kwargs.get('conditions')
        else:
            if 'conditions' in kwargs or 'constraints' in kwargs:
                raise TypeError(
                    'Inconsistent inputs: `constraints`, `conditions`, and `targets` are all equivalent keywords '
                    '- pass all targets as a list to `SciModel`. '
                )
        # setup constraints.
        targets = to_list(targets)
        for i, y in enumerate(targets):
            if not is_constraint(y):
                if is_functional(y):
                    # Case of Data-type constraint.
                    # By default, targets are initialized with Data.
                    targets[i] = Data(y)
                elif isinstance(y, tuple) and \
                        len(y) == 2 and \
                        is_functional(y[0]) and is_functional(y[1]):
                    # Case of Tie-type constraint.
                    targets[i] = Tie(y[0], y[1])
                else:
                    # Not recognised.
                    raise ValueError(
                        'The {}th target entry is not of type `Constraint` or `Functional` - '
                        'received \n ++++++ {} '.format(i, y)
                    )
        # prepare network outputs.
        output_vars = []
        for cond in targets:
            output_vars += cond().outputs
        # prepare loss_functions.
        if not isinstance(loss_func, list):
            loss_func = len(output_vars)*[loss_func]
        # prepare the losses.
        for i, li in enumerate(loss_func):
            if isinstance(li, str):
                loss_func[i] = SciModel.loss_functions(li)
            elif not callable(li):
                raise TypeError(
                    'Please provide a valid loss function from ("mse", "mae") '
                    + "or a callable function for input of tensor types. "
                )
        # Initialize the Model form super class.
        model = Model(
            inputs=input_vars,
            outputs=output_vars,
            **kwargs
        )

        # compile the model.
        loss_weights = [K.variable(1.0) for v in output_vars]

        # Set the variables.
        self._model = model
        self._inputs = inputs
        self._constraints = targets
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._loss_weights = loss_weights
        self._callbacks = {}
        
        # compile the model 
        self.compile()
        
        # set initial state of the model.
        if load_weights_from is not None:
            if os.path.exists(load_weights_from): 
                model.load_weights(load_weights_from)
            else:
                raise Warning("File not found - load_weights_from: {}".format(load_weights_from))

        # Plot to file if requested.
        if plot_to_file is not None:
            plot_model(self._model, to_file=plot_to_file)

    @property
    def model(self):
        return self._model

    @property
    def constraints(self):
        return self._constraints

    @property
    def inputs(self):
        return self._inputs

    def compile(self):
        loss_func = self._loss_func 
        loss_weights = self._loss_weights
        optimizer = self._optimizer

        if isinstance(optimizer, str) and \
                len(optimizer.lower().split("scipy-")) > 1:
            optimizer = GradientObserver(method=optimizer)

        self._model.compile(loss=loss_func,
                            optimizer=optimizer,
                            loss_weights=loss_weights)

    def clear_callbacks(self):
        self._callbacks.clear()

    def load_weights(self, file):
        if os.path.exists(file):
            self.model.load_weights(file)
        else:
            raise ValueError('File not found.')

    def verify_update_constraints(self, constraints):
        ver = []
        for old, new in zip(self._constraints, constraints):
            if old==new and old.sol==new.sol:
                if old.sol is None:
                    ver.append(True)
                else:
                    if all([all(xo==xn) for xo, xn in zip(old.sol, new.sol)]):
                        ver.append(True)
                    else:
                        ver.append(False)
            else:
                ver.append(False)
        return all(ver)

    def __call__(self, *args, **kwargs):
        output = self._model.__call__(*args, **kwargs)
        return output if isinstance(output, list) else [output]

    def save(self, filepath, *args, **kwargs):
        return self._model.save(filepath, *args, **kwargs)

    def save_weights(self, filepath, *args, **kwargs):
        return self._model.save_weights(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        return self._model.load_weights(filepath, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self._model.summary(*args, **kwargs)

    def train(self,
              x_true,
              y_true=None,
              weights=None,
              target_weights=None,
              batch_size=2**6,
              epochs=100,
              learning_rate=0.001,
              adaptive_weights=None,
              adaptive_sample_weights=None,
              log_loss_gradients=None,
              shuffle=True,
              callbacks=[],
              stop_lr_value=1e-8,
              reduce_lr_after=None,
              reduce_lr_min_delta=0.,
              stop_after=None,
              stop_loss_value=1e-8,
              log_parameters=None,
              log_functionals=None,
              log_loss_landscape=None,
              save_weights=None,
              default_zero_weight=0.0,
              validation_data=None,
              **kwargs):
        """Performs the training on the model.

        # Arguments
            x_true: list of `Xs` associated to targets of `Y`.
                Alternatively, you can pass a Sequence object (keras.utils.Sequence).
                Expecting a list of np.ndarray of size (N,1) each,
                with N as the sample size.
            y_true: list of true `Ys` associated to the targets defined during model setup.
                Expecting the same size as list of targets defined in `SciModel`.
                - To impose the targets at specific `Xs` only, pass a tuple of `(ids, y_true)` for that target.
            weights: (np.ndarray) A global sample weight to be applied to samples.
                Expecting an array of shape (N,1), with N as the sample size.
                Default value is `one` to consider all samples equally important.
            target_weights: (list) A weight for each target defined in `y_true`.
            batch_size: (Integer) or 'None'.
                Number of samples per gradient update.
                If unspecified, 'batch_size' will default to 2^6=64.
            epochs: (Integer) Number of epochs to train the model.
                Defaulted to 100.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
            learning_rate: (Tuple/List) (epochs, lrs).
                Expects a list/tuple with a list of epochs and a list or learning rates.
                It linearly interpolates between entries.
                Defaulted to 0.001 with no decay.
                Example:
                    learning_rate = ([0, 100, 1000], [0.001, 0.0005, 0.00001])
            shuffle: Boolean (whether to shuffle the training data).
                Default value is True.
            adaptive_weights: Pass a Dict with the following keys:
                . method: GP or NTK.
                . freq: Freq to update the weights.
                . log_freq: Freq to log the weights and gradients in the history object.
                . beta: The beta parameter in from Gradient Pathology paper.
            log_loss_gradients: Pass a Dict with the following keys:
                . freq: Freq of logs. Defaulted to 100.
                . path: Path to log the gradients.
            callbacks: List of `keras.callbacks.Callback` instances.
            reduce_lr_after: patience to reduce learning rate or stop after certain missed epochs.
                Defaulted to epochs max(10, epochs/10).
            stop_lr_value: stop the training if learning rate goes lower than this value.
                Defaulted to 1e-8.
            reduce_lr_min_delta: min absolute change in total loss value that is considered a successful change.
                Defaulted to 0.001. 
                This values affects number of failed attempts to trigger reduce learning rate based on reduce_lr_after. 
            stop_after: To stop after certain missed epochs. Defaulted to total number of epochs.
            stop_loss_value: The minimum value of the total loss that stops the training automatically. 
                Defaulted to 1e-8.
            log_parameters: Dict object expecting the following keys:
                . parameters: pass list of parameters.
                . freq: pass freq of outputs.
            log_functionals: Dict object expecting the following keys:
                . functionals: List of functionals to log their training history.
                . inputs: The input grid to evaluate the value of each functional.
                          Should be of the same size as the inputs to the model.train.
                . path: Path to the location that the csv files will be logged.
                . freq: Freq of logging the functionals.
            log_loss_landscape: Dict object expecting the following arguments:
                . norm: defaulted to 2.
                . resolution: defaulted to 10.
                . path: Path to the location that the csv files will be logged.
                . trials: Number of trials to pick a path.
            save_weights: (dict) Dict object expecting the following information:
                . path: defaulted to the current path.
                . freq: freq of calling CheckPoint callback.
                . best: If True, only saves the best loss, otherwise save all weights at every `freq` epochs.
            save_weights_to: (file_path) If you want to save the state of the model (at the end of the training).
            save_weights_freq: (Integer) Save weights every N epcohs.
                Defaulted to 0.
            default_zero_weight: a small number for zero sample-weight.

        # Returns
            A Keras 'History' object after performing fitting.
        """
        if isinstance(x_true, Sequence):
            if y_true is not None:
                raise ValueError(
                    'If data_generator is provided, you should not provide y_true.'
                )
            data_generator = x_true

        else:
            if y_true is None:
                raise ValueError('(X_true, Y_true): Please provide proper values for both inputs and targets of SciModel. ')
            # prepare X,Y data.
            x_true = to_list(x_true)
            for i, (x, xt) in enumerate(zip(x_true, self._model.inputs)):
                x_shape = tuple(xt.get_shape().as_list())
                if x.shape != x_shape:
                    try:
                        x_true[i] = x.reshape((-1,) + x_shape[1:])
                    except:
                        print(
                            'Could not automatically convert the inputs to be ' 
                            'of the same size as the expected input tensors. ' 
                            'Please provide inputs of the same dimension as the `Variables`. '
                        )
                        assert False

            y_true = to_list(y_true)
            assert len(y_true)==len(self._constraints), \
                'Miss-match between expected targets (constraints) defined in `SciModel` and ' \
                'the provided `y_true`s - expecting the same number of data points. '

            num_sample = x_true[0].shape[0]
            assert all([x.shape[0]==num_sample for x in x_true[1:]]), \
                'Inconsistent sample size among `Xs`. '
            ids_all = np.arange(0, num_sample)

            if weights is None:
                weights = np.ones(num_sample)
            elif isinstance(weights, np.ndarray):
                if len(weights.shape)!=1 or \
                        weights.shape[0] != num_sample:
                    try:
                        weights = weights.reshape(num_sample)
                    except:
                        raise ValueError(
                            'Input error: `weights` should have dimension 1 with '
                            'the same sample length as `Xs. '
                        )

            sample_weights, y_star = [], []
            for i, yt in enumerate(y_true):
                c = self._constraints[i]
                # verify entry.
                ys, wei = SciModel._prepare_data(
                    c.cond.outputs, to_list(yt),
                    weights if isinstance(weights, np.ndarray) else weights[i],
                    num_sample, default_zero_weight
                )
                # add to the list.
                y_star += ys
                sample_weights += wei

            # create Sequence wrapper for training. 
            data_generator = GeneratorWrapper(
                x_true, y_star, sample_weights, 
                batch_size, shuffle
            )

        # initialize callbacks.
        sci_callbacks = []
        # Learning rate setup.
        K.set_value(self.model.optimizer.lr, 1e-3)
        if reduce_lr_after is None:
            reduce_lr_after = max([10, epochs / 10])
        if stop_after is None:
            stop_after = epochs
        if isinstance(learning_rate, (type(None), float, int)):
            lr_rates = 0.001 if learning_rate is None else learning_rate
            K.set_value(self.model.optimizer.lr, lr_rates)
            lr_scheduler = {
                "scheduler": "default",
                "reduce_lr_after": reduce_lr_after,
                "reduce_lr_min_delta": reduce_lr_min_delta
            }
        elif isinstance(learning_rate, (tuple, list)):
            lr_scheduler = {
                "scheduler": "learning_rate_scheduler",
                "lr_epochs": learning_rate[0],
                "lr_values": learning_rate[1]
            }
        elif isinstance(learning_rate, dict):
            lr_scheduler = learning_rate
        else:
            raise ValueError(
                "learning rate: expecting a `float` or \n  "
                "a tuple/list of two arrays with `epochs` and `learning rates` or \n  "
                "a dictionary with scheduler settings. "
            )
        if 'lr_scheduler' in self._callbacks and learning_rate is None:
            sci_callbacks.append(self._callbacks['lr_scheduler'])
        else:
            sci_callbacks.append(setup_lr_scheduler(lr_scheduler))
            self._callbacks['lr_scheduler'] = sci_callbacks[-1]

        sci_callbacks += [
            k.callbacks.EarlyStopping(monitor="loss", mode='auto', verbose=1,
                                      patience=stop_after, min_delta=1e-9),
            k.callbacks.TerminateOnNaN(),
            EarlyStoppingByLossVal(stop_loss_value),
            EarlyStoppingByLearningRate(stop_lr_value),
            EpochTime()
        ]

        # setup target weights.
        len_inputs = len(to_list(self._model.inputs))
        len_outputs = len(to_list(self._model.outputs))
        if target_weights is not None:
            if not(isinstance(target_weights, list) and
                   len(target_weights) == len_outputs):
                raise ValueError(
                    'Expected a list of weights for the same size as the targets '
                    '- was provided {}'.format(target_weights)
                )
        else:
            target_weights = len_outputs * [1.0]

        # save model.
        model_file_path = None
        if save_weights is not None:
            assert isinstance(save_weights, dict), "pass a dictionary containing `path, freq, best`. "
            if 'path' not in save_weights.keys():
                save_weights_path = os.path.join(os.curdir, "weights")
            else:
                save_weights_path = save_weights['path']
            try:
                if 'best' in save_weights.keys() and \
                        save_weights['best'] is True:
                    model_file_path = save_weights_path + "-best.hdf5"
                    model_check_point = k.callbacks.ModelCheckpoint(
                        model_file_path, monitor='loss', save_weights_only=True, mode='auto',
                        period=10 if 'freq' in save_weights.keys() else save_weights['freq'],
                        save_best_only=True
                    )
                else:
                    self._model.save_weights("{}-start.hdf5".format(save_weights_path))
                    model_file_path = save_weights_path + "-{epoch:05d}-{loss:.3e}.hdf5"
                    model_check_point = k.callbacks.ModelCheckpoint(
                        model_file_path, monitor='loss', save_weights_only=True, mode='auto',
                        period=10 if 'freq' in save_weights.keys() else save_weights['freq'],
                        save_best_only=False
                    )
            except:
                print("\nWARNING: Failed to save model.weights to the provided path: {}\n".format(save_weights_path))
        if model_file_path is not None:
            sci_callbacks.append(model_check_point)

        if isinstance(self._model.optimizer, GradientObserver):
            opt = ScipyOptimizer(self._model)
            opt_fit_func = opt.fit
        else:
            opt_fit_func = self._model.fit

        if adaptive_weights:
            sci_callbacks.append(
                setup_adaptive_weight_callback(
                    adaptive_weights=adaptive_weights,
                    model=self.model,
                    constraints=self.constraints,
                    data_generator=data_generator)
            )
            self._callbacks['adaptive_weights'] = sci_callbacks[-1]
        elif 'adaptive_weights' in self._callbacks:
            sci_callbacks.append(self._callbacks['adaptive_weights'])

        if adaptive_sample_weights:
            # sample_weights = [K.variable(wi) for wi in sample_weights]
            sci_callbacks.append(
                AdaptiveSampleWeight2(
                    self.model, data_generator=data_generator,
                    types=[type(v).__name__ for v in self.constraints],
                    **adaptive_sample_weights
                )
            )
            # loss_gradients = NTKSW.eval_diag_ntk()
            # sample_weights = NTKSW.eval_sample_weights(loss_gradients)
            self._callbacks['adaptive_sample_weights'] = sci_callbacks[-1]

        elif 'adaptive_sample_weights' in self._callbacks:
            sci_callbacks.append(self._callbacks['adaptive_sample_weights'])

        if log_loss_gradients:
            if not isinstance(log_loss_gradients, dict):
                log_loss_gradients = LossGradientHistory.prepare_inputs(log_loss_gradients)
            sci_callbacks.append(
                LossGradientHistory(
                    self.model, data_generator=data_generator,
                    **log_loss_gradients
                )
            )
            self._callbacks['log_loss_gradients'] = sci_callbacks[-1]
        elif 'log_loss_gradients' in self._callbacks:
            sci_callbacks.append(self._callbacks['log_loss_gradients'])

        if log_parameters:
            if not isinstance(log_parameters, dict):
                log_parameters = ParameterHistory.prepare_inputs(log_parameters)
            sci_callbacks.append(
                ParameterHistory(**log_parameters)
            )
            self._callbacks['log_parameters'] = sci_callbacks[-1]
        elif 'log_parameters' in self._callbacks:
            sci_callbacks.append(self._callbacks['log_parameters'])

        if log_functionals:
            if not isinstance(log_functionals, dict):
                log_functionals = FunctionalHistory.prepare_inputs(log_functionals)
            sci_callbacks.append(
                FunctionalHistory(**log_functionals)
            )
            self._callbacks['log_functionals'] = sci_callbacks[-1]
        elif 'log_functionals' in self._callbacks:
            sci_callbacks.append(self._callbacks['log_functionals'])

        if log_loss_landscape:
            if not isinstance(log_loss_landscape, dict):
                log_loss_landscape = LossLandscapeHistory.prepare_inputs(log_loss_landscape)
            sci_callbacks.append(
                LossLandscapeHistory(
                    self.model, data_generator=data_generator,
                    **log_loss_landscape
                )
            )
            self._callbacks['log_loss_landscape'] = sci_callbacks[-1]
        elif 'log_loss_landscape' in self._callbacks:
            sci_callbacks.append(self._callbacks['log_loss_landscape'])

        # training the models.
        history = opt_fit_func(
            data_generator,  # sums to number of samples.
            epochs=epochs,
            callbacks=to_list(sci_callbacks) + to_list(callbacks),
            validation_data=validation_data,
            **kwargs
        )

        if save_weights is not None:
            if 'best' not in save_weights.keys() or \
                    save_weights['best'] is False:
                try:
                    self._model.save_weights("{}-end.hdf5".format(save_weights_path))
                except:
                    print("\nWARNING: Failed to save model.weights to the provided path: {}\n".format(save_weights_path))

        # return the history.
        return history

    def predict(self, xs,
                batch_size=None,
                verbose=0,
                steps=None):
        """ Predict output from network.

        # Arguments
            xs: list of `Xs` associated model.
                Expecting a list of np.ndarray of size (N,1) each,
                with N as the sample size.
            batch_size: defaulted to None.
                Check Keras documentation for more information.
            verbose: defaulted to 0 (None).
                Check Keras documentation for more information.
            steps: defaulted to 0 (None).
                Check Keras documentation for more information.

        # Returns
            List of numpy array of the size of network outputs.

        # Raises
            ValueError if number of `xs`s is different from number of `inputs`.
        """
        xs = to_list(xs)
        if len(xs) != len(self._inputs):
            raise ValueError(
                "Please provide consistent number of inputs as the model is defined: "
                "Expected {} - provided {}".format(len(self._inputs), len(to_list(xs)))
            )
        # To have unified output for postprocessing - limitted support.
        shape_default = [x.shape for x in xs]
        assert all([shape_default[0][0]==x[0] for x in shape_default[1:]])
        # prepare X,Y data.
        for i, (x, xt) in enumerate(zip(xs, self._model.inputs)):
            x_shape = tuple(xt.get_shape().as_list())
            if x.shape != x_shape:
                try:
                    xs[i] = x.reshape((-1,) + x_shape[1:])
                except:
                    print(
                        'Could not automatically convert the inputs to be ' 
                        'of the same size as the expected input tensors. ' 
                        'Please provide inputs of the same dimension as the `Variables`. '
                    )
                    assert False

        y_pred = to_list(self._model.predict(xs, batch_size, verbose, steps))

        # return uniform shapes. 
        if all([shape_default[0]==sd for sd in shape_default[1:]]):
            try:
                y_pred = [y.reshape(shape_default[0]) for y in y_pred]
            except:
                print("Input and output dimensions need re-adjustment for post-processing.")

        # revert back to normal.
        for i, sd in enumerate(shape_default):
            xs[i] = xs[i].reshape(sd)
        xs = unpack_singleton(xs)

        return unpack_singleton(y_pred)

    def eval(self, *args):
        if len(args) == 1:
            x_data = to_list(args[0])
            if len(x_data) != len(self._inputs):
                raise ValueError(
                    "Please provide consistent number of inputs as the model is defined: "
                    "Expected {} - provided {}".format(len(self._inputs), len(x_data))
                )
            if not all([isinstance(xi, np.ndarray) for xi in x_data]):
                raise ValueError("Please provide input data to the network. ")
            return unpack_singleton(self.predict(x_data))

        elif len(args) == 2:
            var_name = args[0]
            if not isinstance(var_name, str):
                raise ValueError("Value Error: Expected a LayerName as the input. ")
            x_data = to_list(args[1])
            new_model = K.function(self._model.inputs, self._model.get_layer(var_name).output)
            if not all([isinstance(xi, np.ndarray) for xi in x_data]):
                raise ValueError("Please provide input data to the network. ")
            return unpack_singleton(new_model(x_data))

    def plot_model(self, *args, **kwargs):
        """ Keras plot_model functionality.
            Refer to Keras documentation for help.
        """
        plot_model(self._model, *args, **kwargs)

    @staticmethod
    def loss_functions(method="mse"):
        """ loss_function returns the callable object to evaluate the loss.

        # Arguments
            method: String.
            - "mse" for `Mean Squared Error` or
            - "mae" for `Mean Absolute Error` or
            - "se" for `Squared Error` or
            - "ae" for `Absolute Error` or
            - "sse" for `Squared Sum Error` or
            - "ase" for `Absolute Sum Error` or
            - "rse" for `Reduce Sum Error`.

        # Returns
            Callable function that gets (y_true, y_pred) as the input and
                returns the loss value as the output.

        # Raises
            ValueError if anything other than "mse" or "mae" is passed.
        """
        if method in ("mse", "mean_squared_error"):
            return lambda y_true, y_pred: K.mean(K.square(y_true - y_pred), axis=-1)
        elif method in ("mae", "mean_absolute_error"):
            return lambda y_true, y_pred: K.mean(K.abs(y_true - y_pred), axis=-1)
        elif method in ("se", "squared_error"):
            return lambda y_true, y_pred: K.sum(K.square(y_true - y_pred), axis=-1)
        elif method in ("ae", "absolute_error"):
            return lambda y_true, y_pred: K.sum(K.abs(y_true - y_pred), axis=-1)
        elif method in ("sse", "squared_sum_error"):
            return lambda y_true, y_pred: K.sum(K.square(K.sum(y_true - y_pred, axis=0, keepdims=True)), axis=-1)
        elif method in ("ase", "absolute_sum_error"):
            return lambda y_true, y_pred: K.sum(K.abs(K.sum(y_true - y_pred, axis=0, keepdims=True)), axis=-1)
        elif method in ("rse", "reduce_sum_error"):
            return lambda y_true, y_pred: K.sum(K.sum(y_true - y_pred, axis=0, keepdims=True), axis=-1)
        elif hasattr(k.losses, method):
            return getattr(k.losses, method)
        else:
            raise ValueError(
                'Supported losses: Keras loss function or (mse, mae, se, ae, sse, ase, rse)'
            )

    @staticmethod
    def _prepare_data(cond_outputs, y_true, global_weights, num_sample, default_zero_weight):
        ys, weis = [], []
        ids_all = np.arange(0, num_sample)
        # prepare sample weight.
        for i, yt in enumerate(to_list(y_true)):
            ids = None
            yc = cond_outputs[i]
            if isinstance(yt, tuple) and len(yt) == 2:
                ids = yt[0].flatten()
                if isinstance(yt[1], np.ndarray):
                    adjusted_yt = yt[1]
                    if ids.size == yt[1].shape[0] and ids.size < num_sample:
                        adjusted_yt = np.zeros((num_sample,)+yt[1].shape[1:])
                        adjusted_yt[ids, :] = yt[1]
                    elif yt[1].shape[0] != num_sample:
                        raise ValueError(
                            'Error in size of the target {}.'.format(i)
                        )
                else:
                    adjusted_yt = yt[1]
                ys.append(adjusted_yt)
            elif isinstance(yt, (np.ndarray, str, float, int, type(None))):
                ys.append(yt)
            else:
                raise ValueError(
                    'Unrecognized entry - please provide a list of `data` or tuples of `(ids, data)`'
                    ' for each target defined in `SciModel`. '
                )
            # Define weights of samples.
            if ids is None:
                ids = ids_all
                wei = global_weights
            else:
                wei = np.zeros(num_sample) + default_zero_weight
                wei[ids] = global_weights[ids]
                wei[ids] *= sum(global_weights)/sum(wei[ids])
            weis.append(wei)
            # preparing targets.
            if isinstance(ys[-1], np.ndarray):
                if not (ys[-1].shape[1:] == k.backend.int_shape(yc)[1:]):
                    try:
                        ys[-1] = ys[-1].reshape((-1,) + k.backend.int_shape(yc)[1:])
                    except (ValueError, TypeError):
                        raise ValueError(
                            'Dimension of expected `y_true` does not match with defined `Constraint`'
                        )
            elif isinstance(ys[-1], str):
                if ys[-1] == 'zero' or ys[-1] == 'zeros':
                    ys[-1] = np.zeros((num_sample, ) + k.backend.int_shape(yc)[1:])
                elif ys[-1] == 'one' or ys[-1] == 'ones':
                    ys[-1] = np.ones((num_sample, ) + k.backend.int_shape(yc)[1:])
                else:
                    raise ValueError(
                        'Unexpected `str` entry - only accepts `zeros` or `ones`.'
                    )
            elif isinstance(ys[-1], (int, float)):
                ys[-1] = np.ones((num_sample, ) + k.backend.int_shape(yc)[1:]) * float(ys[-1])
            elif isinstance(ys[-1], type(None)):
                ys[-1] = np.zeros((num_sample, ) + k.backend.int_shape(yc)[1:])
            else:
                raise ValueError(
                    'Unsupported entry - {} '.format(ys[-1])
                )
            # set undefined ids to zeros.
            if ids.size != num_sample:
                adjusted_ids = np.ones(num_sample, dtype=bool)
                adjusted_ids[ids] = False
                ys[-1][adjusted_ids, :] = 0.0

        return ys, weis

