""" Callbacks module define callbacks for training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.util.tf_export import keras_export

import numpy as np

from .utilities import unpack_singleton, to_list
from .math import tf_gradients


@keras_export('keras.callbacks.EarlyStoppingByLossVal')
class EarlyStoppingByLossVal(Callback):
    """ Callback that terminates the training when loss reaches a small values.
    """
    def __init__(self, value, stop_after=1):
        super(Callback, self).__init__()
        self.value = value
        self.wait = stop_after

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('loss')
        if current < self.value:
            self.wait -= 1
            if self.wait <= 0:
                self.model.stop_training = True
                print("Epoch {:05d}: early stopping at loss value {:0.6e}".format(epoch+1, current))
                print("Revise 'stop_loss_value={:0.12f}' in '.train' if it was not your intent. ".format(self.value))


@keras_export('keras.callbacks.EarlyStoppingByLearningRate')
class EarlyStoppingByLearningRate(Callback):
    """ Callback that terminates the training when learning rate reaches a small values.
    """
    def __init__(self, value):
        super(Callback, self).__init__()
        self.value = value

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('lr')
        if current < self.value:
            self.model.stop_training = True
            print("Epoch {:05d}: early stopping at learning rate {:0.6e}".format(epoch+1, current))
            print("Revise 'stop_lr_value={:0.12f}' in '.train' if it was not your intent. ".format(self.value))


@keras_export('keras.callbacks.GradientPathologyLossWeight')
class GradientPathologyLossWeight(Callback):
    """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
    """
    def __init__(self, model, inputs, targets, weights, beta=0.8, freq=100, log_freq=None, types=None):
        super(Callback, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.weights = weights
        self.loss_grads = []
        for i in range(len(model.outputs)):
            yp = model.outputs[i]
            ys = targets[i]
            ws = weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            gf = tf_gradients(f, model.trainable_weights, unconnected_gradients='zero')
            self.loss_grads.append(
                K.function(model.inputs, gf)
            )
        self.freq = 0 if isinstance(freq, bool) else freq
        self.beta = beta
        assert len(targets) == len(types)
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        if 'PDE' in self.types:
            self.eval_loss_weights = self.eval_loss_weights_m1
        else:
            self.eval_loss_weights = self.eval_loss_weights_m2

    def on_train_begin(self, logs={}):
        # update loss-weights
        loss_gradients = self.eval_loss_gradients()
        self.update_loss_weights(0, loss_gradients)
        # log the weights in the history output.
        logs['adaptive_weights_epochs'] = 0
        logs['adaptive_weights'] = [K.get_value(wi) for wi in self.model.loss_weights]
        # log norm2 of gradients
        logs['loss_gradients'] = [np.linalg.norm(lgi) for lgi in loss_gradients]

    def on_epoch_end(self, epoch, logs={}):
        loss_gradients = None
        if self.freq > 0 and (epoch + 1) % self.freq == 0:
            loss_gradients = self.eval_loss_gradients()
            self.update_loss_weights(epoch, loss_gradients)
        # log gradient values
        if loss_gradients or (self.log_freq > 0 and (epoch + 1) % self.log_freq == 0):
            if loss_gradients is None:
                loss_gradients = self.eval_loss_gradients()
            # log the weights in the history output.
            logs['adaptive_weights_epochs'] = epoch
            logs['adaptive_weights'] = [
                K.get_value(wi) for wi in self.model.loss_weights
            ]
            # log norm2 of gradients
            logs['loss_gradients'] = [
                np.linalg.norm(lgi) for lgi in loss_gradients
            ]

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi, trg in zip(self.loss_grads, self.targets):
            updated_grads.append(
                np.concatenate(
                    [np.abs(wg).flatten() for wg in lgi(self.inputs)]
                )
            )
        return updated_grads

    def eval_loss_weights_m1(self, updated_grads=None):
        # Method 1: normalization by PDE.
        # eval new gradients
        if updated_grads is None:
            updated_grads = self.eval_loss_gradients()
        # eval max normalization on PDE.
        ref_grad = []
        for type, ws in zip(self.types, updated_grads):
            if type is 'PDE':
                ref_grad.append(ws.max())
        ref_grad = max(ref_grad)
        # mean loss terms
        mean_grad = []
        for type, ws in zip(self.types, updated_grads):
            # if type is not 'PDE':
            ws_mean = ws.std()
            mean_grad += [1.0 if ws_mean == 0. else ws_mean]
        # mean_grad = np.mean(mean_grad)
        # evaluate new weights
        new_weights = []
        for i, type in enumerate(self.types):
            if type == 'PDE':
                new_weights.append(1.0)
            elif type in ('Data', 'Tie'):
                new_weights.append(ref_grad / mean_grad[i])
            else:
                assert 'Incorrect target. '
        return new_weights

    def eval_loss_weights_m2(self, updated_grads=None):
        # Method 1: normalization by PDE.
        # eval new gradients
        if updated_grads is None:
            updated_grads = self.eval_loss_gradients()
        # eval max normalization on PDE.
        ref_grad = []
        for ws in updated_grads:
            ref_grad.append(ws.std())
        normalization_grad = sum(ref_grad)
        # evaluate new weights
        new_weights = []
        for i, type in enumerate(self.types):
            new_weights.append(ref_grad[i] / normalization_grad)
        return new_weights

    def update_loss_weights(self, epoch, updated_grads=None):
        new_weights = self.eval_loss_weights(updated_grads)
        # evaluate new weights
        for i, wi in enumerate(self.model.loss_weights):
            gp_weight = new_weights[i] / K.get_value(wi)
            new_val = (1.0 - self.beta) * K.get_value(wi) + self.beta * gp_weight
            K.set_value(self.model.loss_weights[i], new_val)
        # print updates
        print('\n+ adaptive_weights at epoch {}:'.format(epoch+1),
              [K.get_value(wi) for wi in self.model.loss_weights])


@keras_export('keras.callbacks.ParameterHistory')
class ParameterHistory(Callback):
    """ Callback that traces the evolution of parameters.
    """
    def __init__(self, parameters, freq=100):
        super(Callback, self).__init__()
        self.parameters = parameters
        self.freq = freq

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        if epoch % self.freq == 0:
            # log the weights in the history output.
            logs['parameter_epochs'] = epoch
            for param in to_list(self.parameters):
                logs["parameter_" + param.name] = param.value


@keras_export('keras.callbacks.LossGradientHistory')
class LossGradientHistory(Callback):
    """ Callback that evaluate the gradient of loss terms.
    """
    def __init__(self, model, inputs, targets, weights, freq=100):
        super(Callback, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.weights = weights
        self.loss_grads = []
        for i in range(len(model.outputs)):
            yp = model.outputs[i]
            ys = targets[i]
            ws = weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            gf = tf_gradients(f, model.trainable_weights, unconnected_gradients='zero')
            self.loss_grads.append(
                K.function(model.inputs, gf)
            )
        self.freq = 0 if isinstance(freq, bool) else freq

    def on_epoch_end(self, epoch, logs={}):
        loss_gradients = None
        if self.freq > 0 and epoch % self.freq == 0:
            loss_gradients = self.eval_loss_gradients()
            # log norm2 of gradients
            logs['loss_gradients_epochs'] = epoch
            logs['loss_gradients'] = [
                np.linalg.norm(lgi) for lgi in loss_gradients
            ]

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi, trg in zip(self.loss_grads, self.targets):
            updated_grads.append(
                np.concatenate(
                    [np.abs(wg).flatten() for wg in lgi(self.inputs)]
                )
            )
        return updated_grads

