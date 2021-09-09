""" Callbacks module define callbacks for training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import keras as k
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.util.tf_export import keras_export
from scipy.optimize import Bounds, LinearConstraint, minimize

import numpy as np

from .utilities import unpack_singleton, to_list, get_log_path
from .utilities import append_to_bib
from .math import tf_gradients
from time import time


@keras_export('keras.callbacks.EpochTime')
class EpochTime(Callback):
    """ Callback that records the training time.
    """
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)
        logs['time'] = self.times[-1]


@keras_export('keras.callbacks.EarlyStoppingByLossVal')
class EarlyStoppingByLossVal(Callback):
    """ Callback that terminates the training when loss reaches a small values.
    """
    def __init__(self, value, stop_after=1):
        super(EarlyStoppingByLossVal, self).__init__()
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
        super(EarlyStoppingByLearningRate, self).__init__()
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
    def __init__(self, model, data_generator,
                 beta=0.1, freq=100, log_freq=None,
                 types=None,
                 alpha=0.0,
                 min_max=None,
                 **kwargs):
        super(GradientPathologyLossWeight, self).__init__()
        append_to_bib(["wang2020gp"])
        # generate samples. 
        self.inputs, self.targets, self.weights = data_generator[0]
        # eval loss and gradients.
        self.losses = []
        self.loss_grads = []
        for i in range(len(model.outputs)):
            # fixed batch size
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            self.losses.append(K.function(model.inputs, f))
            gf = tf_gradients(f, model.trainable_weights, unconnected_gradients='zero')
            self.loss_grads.append([K.function(model.inputs, gf)])
        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.beta = beta
        assert len(self.targets) == len(types)
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        if 'PDE' in self.types:
            self.eval_loss_weights = self.eval_loss_weights_m1
        else:
            self.eval_loss_weights = self.eval_loss_weights_m2
        self.alpha = alpha
        if alpha>0.:
            self.base_losses = [1. if li==0. else li for li in self.eval_losses()]
        else:
            self.base_losses = None
        if min_max is None:
            self.min_max = [-np.Inf, np.Inf]
        else:
            assert len(min_max)==2
            self.min_max = min_max

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.freq == 0:
            self.update(epoch)

    def update(self, epoch):
        losses = self.eval_losses() if self.base_losses else None
        loss_gradients = self.eval_loss_gradients()
        self.update_loss_weights(epoch, loss_gradients, losses)
        self.update_loss_gradients(epoch, loss_gradients)

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        for i, wi in enumerate(self.loss_weights):
            logs[f'loss_weight_{i}'] = wi
        # log norm2 of gradients
        for i, gi in enumerate(self.loss_gradients):
            logs[f'loss_gradient_{i}'] = gi
    
    def eval_losses(self):
        return [l(self.inputs) for l in self.losses]

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi in self.loss_grads:
            losses = []
            for lgij in lgi:
                losses.append(
                    np.concatenate(
                        [np.abs(wg).flatten() for wg in lgij(self.inputs)]
                    )
                )
            updated_grads.append(np.concatenate(losses))
        return updated_grads

    def eval_loss_weights_m1(self, updated_grads):
        # Method 1: normalization by PDE.
        # eval max normalization on PDE.
        ref_grad = []
        for type, ws in zip(self.types, updated_grads):
            if type == "PDE":
                ref_grad.append(ws.max())
        ref_grad = max(ref_grad)
        # mean loss terms
        mean_grad = []
        for type, ws in zip(self.types, updated_grads):
            # if type is not 'PDE':
            ws_mean = ws.mean()
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

    def eval_loss_weights_m2(self, updated_grads):
        # Method 1: normalization by PDE.
        # eval max normalization on PDE.
        ref_grad = []
        for ws in updated_grads:
            ref_grad.append(ws.max())
        normalization_grad = max(ref_grad)
        # mean loss terms
        mean_grad = [ws.mean() for ws in updated_grads]
        # evaluate new weights
        new_weights = []
        for i, type in enumerate(self.types):
            ref_grad_i = mean_grad[i] if mean_grad[i]>0. else np.mean(mean_grad)
            new_weights.append(normalization_grad / ref_grad_i)
        return new_weights

    def update_loss_weights(self, epoch, updated_grads, updated_losses):
        new_weights = self.eval_loss_weights(updated_grads)
        n_task = len(new_weights)
        
        if updated_losses:
            new_scores = [li/l0 for li, l0 in zip(updated_losses, self.base_losses)]
            new_scores = [(li/np.mean(new_scores))**self.alpha for li in new_scores]
        else:
            new_scores = n_task*[1.]

        old_weights = [K.get_value(wi) for wi in self.model.loss_weights]
        gp_weights = [new_weights[i]/old_weights[i]*new_scores[i] for i in range(n_task)]
        # check for limiting weights.
        for i, wi in enumerate(gp_weights):
            if wi < self.min_max[0]:
                gp_weights[i] = self.min_max[0]
            elif wi > self.min_max[1]:
                gp_weights[i] = self.min_max[1]
        # normalizing weights.
        norm = len(gp_weights)/sum(gp_weights)
        gp_weights = [gpi*norm for gpi in gp_weights]

        self.loss_weights = []
        # evaluate new weights
        for i, wi in enumerate(old_weights):
            new_val = (1.0 - self.beta) * wi + self.beta * gp_weights[i]
            K.set_value(self.model.loss_weights[i], new_val)
            self.loss_weights.append(new_val)
        # print updates
        print('\n+ adaptive_weights at epoch {}:'.format(epoch+1), self.loss_weights)

    def update_loss_gradients(self, epoch, updated_grads):
        self.loss_gradients = [np.linalg.norm(lgi) for lgi in updated_grads]

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'GP'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.GradNormLossWeight')
class GradNormLossWeight(Callback):
    """ Callback that evaluate the adaptive weights based on the GradNorm algorithm by Chen et al.
    """
    def __init__(self, model, data_generator,
                 beta=0.1, freq=100, log_freq=None, types=None,
                 alpha=0.0, min_max=None, **kwargs):
        super(GradNormLossWeight, self).__init__()
        append_to_bib("chen2018gradnorm")
        # generate samples.
        self.inputs, self.targets, self.weights = data_generator[0]
        # eval loss and gradients.
        self.losses = []
        self.loss_grads = []
        for i in range(len(model.outputs)):
            # fixed batch size
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            self.losses.append(K.function(model.inputs, f))
            gf = tf_gradients(f, model.trainable_weights, unconnected_gradients='zero')
            self.loss_grads.append([K.function(model.inputs, gf)])
        self.freq = np.Inf if (freq == True or freq == 0) else freq
        self.beta = beta
        assert len(self.targets) == len(types)
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        self.alpha = alpha
        if alpha > 0.:
            self.base_losses = [1. if li == 0. else li for li in self.eval_losses()]
        else:
            self.base_losses = None
        if min_max is None:
            self.min_max = [-np.Inf, np.Inf]
        else:
            assert len(min_max) == 2
            self.min_max = min_max

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.freq == 0:
            self.update(epoch)

    def update(self, epoch):
        losses = self.eval_losses() if self.base_losses else None
        loss_gradients = self.eval_loss_gradients()
        self.update_loss_weights(epoch, loss_gradients, losses)
        self.update_loss_gradients(epoch, loss_gradients)

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        for i, wi in enumerate(self.loss_weights):
            logs[f'loss_weight_{i}'] = wi
        # log norm2 of gradients
        for i, gi in enumerate(self.loss_gradients):
            logs[f'loss_gradient_{i}'] = gi

    def eval_losses(self):
        return [l(self.inputs) for l in self.losses]

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi in self.loss_grads:
            losses = []
            for lgij in lgi:
                losses.append(
                    np.concatenate(
                        [np.abs(wg).flatten() for wg in lgij(self.inputs)]
                    )
                )
            updated_grads.append(np.concatenate(losses))
        return updated_grads

    def eval_loss_weights(self, updated_grads):
        # get old weights
        old_weights = [K.get_value(wi) for wi in self.model.loss_weights]
        # eval gradients.
        ref_grad = []
        for wi, gi in zip(old_weights, updated_grads):
            ref_grad.append(wi*np.linalg.norm(gi, 2))
        normalization_grad = np.mean(ref_grad)
        # evaluate new weights
        new_weights = []
        for i, type in enumerate(self.types):
            ref_grad_i = ref_grad[i]
            if ref_grad_i==0.:
                ref_grad_i = normalization_grad
            new_weights.append(normalization_grad / ref_grad_i)
        return new_weights

    def update_loss_weights(self, epoch, updated_grads, updated_losses):
        new_weights = self.eval_loss_weights(updated_grads)
        n_task = len(new_weights)

        if updated_losses:
            new_scores = [li / l0 for li, l0 in zip(updated_losses, self.base_losses)]
            mean_scores = np.mean(new_scores)
            new_scores = [(li / mean_scores) ** self.alpha for li in new_scores]
        else:
            new_scores = n_task * [1.]

        old_weights = [K.get_value(wi) for wi in self.model.loss_weights]
        gp_weights = [new_weights[i] * new_scores[i] for i in range(n_task)]

        # check for limiting weights.
        for i, wi in enumerate(gp_weights):
            if wi < self.min_max[0]:
                gp_weights[i] = self.min_max[0]
            elif wi > self.min_max[1]:
                gp_weights[i] = self.min_max[1]

        # normalizing weights.
        norm = len(gp_weights) / sum(gp_weights)
        gp_weights = [gpi * norm for gpi in gp_weights]

        self.loss_weights = []
        # evaluate new weights
        for i, wi in enumerate(old_weights):
            new_val = (1.0 - self.beta) * wi + self.beta * gp_weights[i]
            K.set_value(self.model.loss_weights[i], new_val)
            self.loss_weights.append(new_val)
        # print updates
        print('\n+ adaptive_weights at epoch {}:'.format(epoch + 1), self.loss_weights)

    def update_loss_gradients(self, epoch, updated_grads):
        self.loss_gradients = [np.linalg.norm(lgi, 2) for lgi in updated_grads]

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'GN'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.InverseDirichletLossWeight')
class InverseDirichletLossWeight(Callback):
    """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
    """

    def __init__(self, model, data_generator,
                 beta=0.5, freq=100, log_freq=None,
                 hessian=False, types=None,
                 alpha=0.0, min_max=None, **kwargs):
        super(InverseDirichletLossWeight, self).__init__()
        append_to_bib(["maddu2021inversedirichlet"])
        # generate samples.
        self.inputs, self.targets, self.weights = data_generator[0]
        # eval loss and gradients.
        self.losses = []
        self.loss_grads = []
        for i in range(len(model.outputs)):
            # fixed batch size
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            self.losses.append(K.function(model.inputs, f))
            gf = tf_gradients(f, model.trainable_weights, unconnected_gradients='zero')
            if hessian is True:
                g2f = []
                for gfi in gf:
                    g2fi = tf_gradients(gfi, model.trainable_weights, unconnected_gradients='zero')
                    g2f.append(K.function(model.inputs, g2fi))
                self.loss_grads.append(g2f)
            else:
                self.loss_grads.append([K.function(model.inputs, gf)])
        self.freq = np.Inf if (freq == True or freq == 0) else freq
        self.beta = beta
        assert len(self.targets) == len(types)
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        self.alpha = alpha
        if alpha > 0.:
            self.base_losses = [1. if li==0. else li for li in self.eval_losses()]
        else:
            self.base_losses = None
        if min_max is None:
            self.min_max = [-np.Inf, np.Inf]
        else:
            assert len(min_max) == 2
            self.min_max = min_max

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.freq == 0:
            self.update(epoch)

    def update(self, epoch):
        losses = self.eval_losses() if self.base_losses else None
        loss_gradients = self.eval_loss_gradients()
        self.update_loss_weights(epoch, loss_gradients, losses)
        self.update_loss_gradients(epoch, loss_gradients)

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        for i, wi in enumerate(self.loss_weights):
            logs[f'loss_weight_{i}'] = wi
        # log norm2 of gradients
        for i, gi in enumerate(self.loss_gradients):
            logs[f'loss_gradient_{i}'] = gi

    def eval_losses(self):
        return [l(self.inputs) for l in self.losses]

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi in self.loss_grads:
            losses = []
            for lgij in lgi:
                losses.append(
                    np.concatenate(
                        [np.abs(wg).flatten() for wg in lgij(self.inputs)]
                    )
                )
            updated_grads.append(np.concatenate(losses))
        return updated_grads

    def eval_loss_weights(self, updated_grads):
        # Method 1: normalization by PDE.
        # eval max normalization on PDE.
        ref_grad = []
        for ws in updated_grads:
            ref_grad.append(ws.std())
        normalization_grad = max(ref_grad)
        # evaluate new weights
        new_weights = []
        for i, type in enumerate(self.types):
            ref_grad_i = ref_grad[i] if ref_grad[i]>0. else np.mean(ref_grad)
            new_weights.append(normalization_grad / ref_grad_i)
        return new_weights

    def update_loss_weights(self, epoch, updated_grads, updated_losses):
        new_weights = self.eval_loss_weights(updated_grads)
        n_task = len(new_weights)

        if updated_losses:
            new_scores = [li / l0 for li, l0 in zip(updated_losses, self.base_losses)]
            new_scores = [(li / np.mean(new_scores)) ** self.alpha for li in new_scores]
        else:
            new_scores = n_task * [1.]

        old_weights = [K.get_value(wi) for wi in self.model.loss_weights]
        gp_weights = [new_weights[i] * new_scores[i] for i in range(n_task)]
        # check for limiting weights.
        for i, wi in enumerate(gp_weights):
            if wi < self.min_max[0]:
                gp_weights[i] = self.min_max[0]
            elif wi > self.min_max[1]:
                gp_weights[i] = self.min_max[1]
        # normalizing weights.
        norm = len(gp_weights) / sum(gp_weights)
        gp_weights = [gpi * norm for gpi in gp_weights]

        self.loss_weights = []
        # evaluate new weights
        for i, wi in enumerate(old_weights):
            new_val = (1.0 - self.beta) * wi + self.beta * gp_weights[i]
            K.set_value(self.model.loss_weights[i], new_val)
            self.loss_weights.append(new_val)
        # print updates
        print('\n+ adaptive_weights at epoch {}:'.format(epoch + 1), self.loss_weights)

    def update_loss_gradients(self, epoch, updated_grads):
        self.loss_gradients = [np.linalg.norm(lgi) for lgi in updated_grads]

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'ID'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.LossGradientHistory')
class LossGradientHistory(Callback):
    """ Callback that evaluate the gradient of loss terms.
    """
    def __init__(self, model, data_generator, path=None, freq=100, hessian=False):
        super(LossGradientHistory, self).__init__()
        # generate samples. 
        self.inputs, self.targets, self.weights = data_generator[0]

        self.loss_grads = []
        self.loss_hessians = [] if hessian else None
        for i in range(len(model.outputs)):
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            gf = tf_gradients(f, model.trainable_weights, unconnected_gradients='zero')
            self.loss_grads.append(
                K.function(model.inputs, gf)
            )
            if hessian:
                g2f = []
                for gfi in gf:
                    g2fi = tf_gradients(gfi, model.trainable_weights, unconnected_gradients='zero')
                    g2f.append(K.function(model.inputs, g2fi))
                self.loss_hessians.append(g2f)
        self.path = get_log_path(path, 'loss-')
        self.freq = np.Inf if (freq==True or freq==0) else freq

    def on_epoch_end(self, epoch, logs={}):
        loss_gradients = None
        if self.freq > 0 and epoch % self.freq == 0:
            loss_gradients = self.eval_loss_gradients()
            # output full gradient to the log path.
            for i, lg in enumerate(loss_gradients):
                np.savetxt(
                    self.path + "gradient-history-L{}-epoch{}.csv".format(i+1, epoch+1),
                    lg.reshape(-1, 1),
                    delimiter=','
                )
            # eval the hessian matrix.
            if self.loss_hessians:
                loss_hessians = self.eval_loss_hessians()
                for i, lh in enumerate(loss_hessians):
                    np.savetxt(
                        self.path + "hessian-history-L{}-epoch{}.csv".format(i+1, epoch+1),
                        lh.reshape(-1, 1),
                        delimiter=','
                    )

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi in self.loss_grads:
            updated_grads.append(
                np.concatenate(
                    [wg.flatten() for wg in lgi(self.inputs)]
                )
            )
        return updated_grads

    def eval_loss_hessians(self):
        # eval new gradients
        updated_hessians = []
        for lhi in self.loss_hessians:
            loss_hessians = []
            for lhij in lhi:
                wlhij = lhij(self.inputs)
                loss_hessians.append(
                    np.concatenate([wg.flatten() for wg in wlhij])
                )
            # append for each layer.
            updated_hessians.append(np.concatenate(loss_hessians))
        return updated_hessians

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.MTLLossWeight')
class MTLLossWeight(Callback):
    """ Callback that evaluate the gradient of loss terms.
    """

    def __init__(self, model, data_generator, freq=100, **kwargs):
        super(MTLLossWeight, self).__init__()
        append_to_bib("sener2018multi")
        # generate samples.
        self.inputs, self.targets, self.weights = data_generator[0]

        self.loss_grads = []
        self.parameters = model.trainable_weights
        for i in range(len(model.outputs)):
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            gf = tf_gradients(f, self.parameters, unconnected_gradients='zero')
            self.loss_grads.append(
                K.function(model.inputs, gf)
            )
        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.hist_gradient = []
        self.hist_parameters = []
        self.loss_weights = len(model.outputs) * [1.]
        self.loss_hessians = len(model.outputs) * [0.]

    def on_epoch_begin(self, epoch, logs={}):
        # pass
        loss_gradients = None
        if epoch % self.freq == 0:
            grads = self.eval_loss_gradients()
            num_Ts = len(grads)

            M = np.array([[np.dot(gi, gj) for gj in grads] for gi in grads])
            alpha0 = np.ones(num_Ts)

            f = lambda x: 0.5*np.matmul(M, x).dot(x)
            jac = lambda x: np.matmul(M, x)
            hess = lambda x: M

            bounds = Bounds(np.zeros(num_Ts), np.ones(num_Ts)*num_Ts)
            constraint = LinearConstraint(np.ones(num_Ts), num_Ts, num_Ts)

            alpha = minimize(
                f, alpha0, method='trust-constr',
                jac=jac, hess=hess, bounds=bounds, constraints=constraint
            )

            self.loss_weights = alpha.x
            self.loss_gradients = [np.linalg.norm(hs) for hs in grads]

            for i, wi in enumerate(self.model.loss_weights):
                gp_weight = self.loss_weights[i]
                new_val = gp_weight  # (1.0 - self.beta) * K.get_value(wi) + self.beta * gp_weight
                K.set_value(self.model.loss_weights[i], new_val)

            # print updates
            print('\n+ adaptive_weights at epoch {}:'.format(epoch + 1), self.loss_weights)

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        logs['adaptive_weights'] = self.loss_weights
        # log norm2 of gradients
        logs['loss_gradients'] = self.loss_gradients

    def eval_loss_gradients(self):
        # eval new gradients
        updated_grads = []
        for lgi in self.loss_grads:
            updated_grads.append(
                np.concatenate(
                    [wg.flatten() for wg in lgi(self.inputs)]
                )
            )
        return updated_grads

    def eval_parameters(self):
        return np.concatenate([K.get_value(wi).flatten() for wi in self.parameters])

    def eval_loss_hessians(self):
        # eval new gradients
        updated_hessians = []
        for lhi in self.loss_hessians:
            loss_hessians = []
            for lhij in lhi:
                wlhij = lhij(self.inputs)
                loss_hessians.append(
                    np.concatenate([wg.flatten() for wg in wlhij])
                )
            # append for each layer.
            updated_hessians.append(np.concatenate(loss_hessians))
        return updated_hessians

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.ScoreLossWeight')
class ScoreLossWeight(Callback):
    """ Callback that evaluate the gradient of loss terms.
    """

    def __init__(self, model, data_generator, freq=100, 
                 alpha=1., beta=0.1, min_max=None, **kwargs):
        super(ScoreLossWeight, self).__init__()
        # generate samples.
        self.inputs, self.targets, self.weights = data_generator[0]

        self.losses = []
        for i in range(len(model.outputs)):
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            f = tf.reduce_mean(
                model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            self.losses.append(K.function(model.inputs, f))

        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.alpha = alpha
        self.beta = beta
        self.base_losses = [1. if li == 0. else li for li in self.eval_losses()]
        if min_max is None:
            self.min_max = [-np.Inf, np.Inf]
        else:
            assert len(min_max) == 2
            self.min_max = min_max

    def on_epoch_begin(self, epoch, logs={}):
        # pass
        if epoch % self.freq == 0:
            losses = self.eval_losses()
            n_task = len(losses)

            scores = [li/l0 for li, l0 in zip(losses, self.base_losses)]
            scores = [(li/np.mean(scores))**self.alpha for li in scores]
            norm = n_task/sum(scores)
            new_weights = [si*norm for si in scores]

            # check for limiting weights.
            for i, wi in enumerate(new_weights):
                if wi < self.min_max[0]:
                    new_weights[i] = self.min_max[0]
                elif wi > self.min_max[1]:
                    new_weights[i] = self.min_max[1]

            old_weights = [K.get_value(wi) for wi in self.model.loss_weights]
            
            self.loss_weights = []
            # evaluate new weights
            for i, wi in enumerate(old_weights):
                new_val = (1.0 - self.beta) * wi + self.beta * new_weights[i]
                K.set_value(self.model.loss_weights[i], new_val)
                self.loss_weights.append(new_val)

            # print updates
            print('\n+ adaptive_weights at epoch {}:'.format(epoch+1), self.loss_weights)

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        for i, wi in enumerate(self.loss_weights):
            logs[f'loss_weight_{i}'] = wi

    def eval_losses(self):
        return [l(self.inputs) for l in self.losses]

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.ParameterHistory')
class ParameterHistory(Callback):
    """ Callback that traces the evolution of parameters.
    """
    def __init__(self, parameters, freq=100):
        super(ParameterHistory, self).__init__()
        self.parameters = parameters
        self.freq = freq

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        if epoch % self.freq == 0:
            # log the weights in the history output.
            # logs['parameter_epochs'] = epoch
            for param in to_list(self.parameters):
                logs[param.name] = param.value

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        if len(args) == 1:
            kwargs['parameters'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.FunctionalHistory')
class FunctionalHistory(Callback):
    """ Callback that traces the evolution of field variables.
    """
    def __init__(self, functionals=[], inputs=None, sci_model=None, path=None, freq=100):
        super(FunctionalHistory, self).__init__()
        self.functionals = to_list(functionals)
        self.inputs = inputs
        self.sci_model = sci_model
        self.path = get_log_path(path, "functional-history")
        self.freq = freq

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        if epoch % self.freq == 0:
            logs['parameter_epochs'] = epoch
            self.log_functionals(epoch+1)

    def on_train_end(self, logs={}):
        self.log_functionals('-end')

    def log_functionals(self, epoch=None):
        # log the weights in the history output.
        for field in self.functionals:
            if self.sci_model is None:
                field_value = field.eval(self.inputs)
            else:
                field_value = field.eval(self.sci_model, self.inputs)
            np.savetxt(
                self.path + "-{}-epoch{}.csv".format(field.name, epoch),
                field_value,
                delimiter=','
            )

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        if len(args) == 1:
            kwargs['functionals'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


class LossLandscapeHistory(Callback):
    """ Callback that traces the evolution of loss function.
    """
    def __init__(self, model, data_generator,
                 norm=2, resolution=11, layer_wise=True,
                 path=None, trials=1):
        super(LossLandscapeHistory, self).__init__()
        append_to_bib("li2017visualizing")
        # generate samples. 
        inputs, targets, weights = data_generator[0]
        self._model = model
        self._inputs = inputs
        self._layers = [layer for layer in model._layers if layer.weights]
        self._weights_size = 0
        for layer in self._layers:
            for w in layer.weights:
                if not w.trainable:
                    continue
                self._weights_size += np.prod(w.shape)
        # collect initial value of the weights.
        self._norm = lambda xs: np.linalg.norm(xs, norm)
        self._resolution = resolution
        self._layer_wise = layer_wise
        self._weight_norm = [self._norm(np.concatenate(self._collect_weights()))]
        self._path = get_log_path(path, "loss-landscape-history")
        self._trials = trials

        f = None
        for i in range(len(model.outputs)):
            yp = model.outputs[i]
            ys = targets[i]
            ws = weights[i]
            fi = tf.reduce_mean(
                    model.loss_functions[i](ys, yp, sample_weight=ws)
            )
            if f is None:
                f = fi
            else:
                f += fi
        self._loss = K.function(model.inputs, f)
        self._loss_value = [self._loss(inputs)]

    def _update_weights(self, x):
        k = -1
        for layer in self._layers:
            w_list = []
            w_trainable = [w.trainable for w in layer.weights]
            batch_update = False not in w_trainable
            for w in layer.weights:
                if not w.trainable:
                    continue
                k += 1
                shape = w.get_shape()
                value = np.array(x[k]).reshape(shape)
                if batch_update:
                    w_list.append(value)
                else:
                    K.set_value(w, value)
            if batch_update:
                layer.set_weights(w_list)

    def _collect_weights(self):
        x_values = []
        for layer in self._layers:
            w_trainable = [w.trainable for w in layer.weights]
            for var, trainable in zip(layer.get_weights(), w_trainable):
                if trainable:
                    x_values.append(var.reshape(-1))
        return x_values

    def on_epoch_end(self, epoch, logs={}):
        # if epoch % self._freq != 0:
        #     return
        x_trained = self._collect_weights()
        self._weight_norm.append(self._norm(np.concatenate(x_trained)))
        self._loss_value.append(self._loss(self._inputs))
        logs['norm-loss-weights'] = self._weight_norm[-1]

    def on_train_end(self, logs={}):
        x_trained = self._collect_weights()
        x_sizes = [x.size for x in x_trained]
        num_param = sum(x_sizes)
        # different trials to check consistency of landscape plot.
        for trial in range(self._trials):
            n0 = np.split(np.random.standard_normal(num_param), np.cumsum(x_sizes))
            n1 = np.split(np.random.standard_normal(num_param), np.cumsum(x_sizes))
            # normalize layer wise
            if self._layer_wise:
                n0 = [ni/self._norm(ni) for ni in n0]
                n1 = [ni/self._norm(ni) for ni in n1]
            # normalize globally
            n0_norm, n1_norm = [self._norm(np.concatenate(ni)) for ni in [n0, n1]]
            n0 = [ni/n0_norm for ni in n0]
            n1 = [ni/n1_norm for ni in n1]

            delta_weights = 2.0 * abs(self._weight_norm[-1] - self._weight_norm[0])
            loss_values = np.zeros((self._resolution**2, 3))

            k = 0
            for i, l0 in enumerate(np.linspace(-delta_weights, delta_weights, self._resolution)):
                # l0_weight = self._norm(np.concatenate([xi + n0i*l0 for xi,n0i in zip(x_trained, n0)]))
                for j, l1 in enumerate(np.linspace(-delta_weights, delta_weights, self._resolution)):
                    # l1_weight = self._norm(np.concatenate([xi + n1i*l1 for xi,n1i in zip(x_trained, n1)]))
                    test_weights = [xi + n0i*l0 + n1i*l1 for xi,n0i,n1i in zip(x_trained, n0, n1)]
                    self._update_weights(test_weights)
                    loss_values[k, :] = [l0, l1, self._loss(self._inputs)]
                    k += 1

            # save the calculations
            np.savetxt(
                # self._path + "-epoch{}.csv".format(epoch + 1),
                self._path + "-landscape{}.csv".format("" if self._trials==1 else trial),
                loss_values,
                delimiter=','
            )

        # save weight history
        np.savetxt(
            self._path + "-trace.csv",
            [[n0i, n0i, li] for n0i, li in zip(self._weight_norm, self._loss_value)],
            delimiter=','
        )
        # reset the weights
        self._update_weights(x_trained)

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], int):
            kwargs['resolution'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs



@keras_export('keras.callbacks.NTKLossWeight')
class NTKLossWeight(Callback):
    """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
    """
    def __init__(self, model, data_generator,
                 beta=0.9, freq=100, log_freq=None,
                 alpha=0., types=None, min_max=None, **kwargs):
        super(NTKLossWeight, self).__init__()
        # print(f"{beta = }, {freq = }, {alpha = }, {min_max = }")
        append_to_bib("wang2020ntk")
        # generate samples.
        self.inputs, self.targets, self.weights = data_generator[0]
        # eval loss and gradients.
        self.losses, self.loss_grads = [], []
        for i in range(len(model.outputs)):
            # fixed batch size
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            # eval losses
            if alpha>0.:
                f = tf.reduce_mean(
                    model.loss_functions[i](ys, yp, sample_weight=ws)
                )
                self.losses.append(K.function(model.inputs, f))
            # eval gradients
            gf = tf_gradients(yp, model.trainable_weights, unconnected_gradients='zero')
            gf_dot_gf = tf.add_n([tf.reduce_sum(tf.multiply(gf_i, gf_i)) for gf_i in gf])
            self.loss_grads.append(K.function(model.inputs, gf_dot_gf))
        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.beta = beta
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        assert len(model.outputs) == len(types)
        self.alpha = alpha
        if alpha>0.:
            self.base_losses = self.eval_losses()
        else:
            self.base_losses = None
        if min_max is None:
            self.min_max = [-np.Inf, np.Inf]
        else:
            assert len(min_max) == 2
            self.min_max = min_max

    @staticmethod
    def collect_weights(model):
        x_values = []
        for weights in model.trainable_weights:
            x_values.append(tf.reshape(weights, (-1,)))
        return tf.concat(x_values, axis=0)

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.freq == 0:
            losses = self.eval_losses() if self.base_losses else None
            loss_gradients = self.eval_diag_ntk()
            self.update_loss_weights(epoch, loss_gradients, losses)
            self.update_loss_gradients(epoch, loss_gradients)

    def on_epoch_end(self, epoch, logs={}):
        # log gradient values
        for i, wi in enumerate(self.loss_weights):
            logs[f'loss_weight_{i}'] = wi
        # log norm2 of gradients
        for i, gi in enumerate(self.loss_gradients):
            logs[f'loss_gradient_{i}'] = gi

    def eval_diag_ntk(self):
        # eval new gradients
        num_samples = self.inputs[0].shape[0]
        updated_NTK = [np.zeros((num_samples,)) for lgi in self.loss_grads]
        start_time = time()
        for li, lgi in enumerate(self.loss_grads):
            for i in range(num_samples):
                wi = self.weights[li][i]
                if wi == 0.: continue
                xi = [v[i:i + 1, :] for v in self.inputs]
                updated_NTK[li][i] = lgi(xi)
        print("\n+ NTK evaluation time [s]: {} ".format(time() - start_time))
        return updated_NTK

    def eval_losses(self):
        losses = [fi(self.inputs) for fi in self.losses]
        return [1. if li==0. else li for li in losses]

    def eval_loss_weights(self, updated_grads):
        # eval max normalization on PDE.
        ref_grad = []
        for ws in updated_grads:
            ref_grad.append(ws.sum())
        normalization_grad = sum(ref_grad)
        # evaluate new weights
        new_weights = []
        for i, type in enumerate(self.types):
            ref_grad_i = ref_grad[i] if ref_grad[i] != 0. else np.mean(ref_grad)
            new_weights.append(normalization_grad / ref_grad_i)
        return new_weights

    def update_loss_weights(self, epoch, updated_grads, updated_losses):
        new_weights = self.eval_loss_weights(updated_grads)
        # print(f"{new_weights = }")
        n_task = len(new_weights)

        if updated_losses:
            new_scores = [li / l0 for li, l0 in zip(updated_losses, self.base_losses)]
            mean_scores = np.mean(new_scores)
            new_scores = [(li / mean_scores) ** self.alpha for li in new_scores]
        else:
            new_scores = n_task * [1.]
        # print(f"{self.base_losses = }")
        # print(f"{updated_losses = }")
        # print(f"{new_scores = }")

        old_weights = [K.get_value(wi) for wi in self.model.loss_weights]
        ntk_weights = [new_weights[i] * new_scores[i] for i in range(n_task)]
        # print(f"{ntk_weights = }")

        # check for limiting weights.
        for i, wi in enumerate(ntk_weights):
            if wi < self.min_max[0]:
                ntk_weights[i] = self.min_max[0]
            elif wi > self.min_max[1]:
                ntk_weights[i] = self.min_max[1]

        # normalizing weights.
        norm = n_task / sum(ntk_weights)
        ntk_weights = [wi * norm for wi in ntk_weights]

        # print(f"{norm = }")
        # print(f"{ntk_weights = }")

        self.loss_weights = []
        # evaluate new weights
        for i, wi in enumerate(old_weights):
            new_val = (1-self.beta) * wi + self.beta * ntk_weights[i]
            K.set_value(self.model.loss_weights[i], new_val)
            self.loss_weights.append(new_val)

        # print updates
        print(f'\n+ adaptive_weights at epoch {epoch+1}:', self.loss_weights)

    def update_loss_gradients(self, epoch, updated_grads):
        self.loss_gradients = [np.linalg.norm(lgi, 2) for lgi in updated_grads]

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'NTK'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


#
# @keras_export('keras.callbacks.NTKLossWeight')
# class NTKLossWeight(Callback):
#     """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
#     """
#     def __init__(self, model, data_generator,
#                  beta=0.1, freq=100, log_freq=None,
#                  hessian=False, types=None, **kwargs):
#         super(NTKLossWeight, self).__init__()
#         append_to_bib("wang2020ntk")
#         # generate samples.
#         inputs, targets, weights = data_generator.get_data()
#         # limit number of samples for performance concerns.
#         MAX_SAMPLE = 10000
#         if inputs[0].shape[0] > MAX_SAMPLE:
#             sample_ids = np.random.choice(inputs[0].shape[0], MAX_SAMPLE, replace=False)
#             self.inputs = [x[sample_ids] for x in inputs]
#             self.targets = [y[sample_ids] for y in targets]
#             self.weights = [w[sample_ids] for w in weights]
#         else:
#             self.inputs = inputs
#             self.targets = targets
#             self.weights = weights
#         # eval loss and gradients.
#         self.loss_grads = []
#         for i in range(len(model.outputs)):
#             # fixed batch size
#             yp = model.outputs[i]
#             ws = self.weights[i]
#             gf = tf_gradients(yp, model.trainable_weights, unconnected_gradients='zero')
#             self.loss_grads.append(K.function(model.inputs, gf))
#         self.freq = np.Inf if (freq==True or freq==0) else freq
#         self.beta = beta
#         self.types = types
#         if log_freq is None:
#             self.log_freq = self.freq
#         else:
#             self.log_freq = log_freq
#         assert len(model.outputs) == len(types)
#
#     @staticmethod
#     def collect_weights(model):
#         x_values = []
#         for weights in model.trainable_weights:
#             x_values.append(tf.reshape(weights, (-1,)))
#         return tf.concat(x_values, axis=0)
#
#     def update(self, epoch):
#         loss_gradients = self.eval_diag_ntk()
#         self.update_loss_weights(epoch, loss_gradients)
#         self.update_loss_gradients(epoch, loss_gradients)
#
#     def on_train_begin(self, logs={}):
#         self.update(0)
#
#     def on_epoch_begin(self, epoch, logs={}):
#         if self.freq > 0 and (epoch + 1) % self.freq == 0:
#             self.update(epoch)
#
#     def on_epoch_end(self, epoch, logs={}):
#         # log gradient values
#         logs['adaptive_weights'] = self.loss_weights
#         # log norm2 of gradients
#         logs['loss_gradients'] = self.loss_gradients
#
#     def eval_full_ntk(self):
#         # eval new gradients
#         num_samples = self.inputs[0].shape[0]
#         ntk = [
#             [np.zeros((num_samples,num_samples)) for lgi in self.loss_grads]
#             for lgj in self.loss_grads
#         ]
#         for i in range(num_samples):
#             xi = [v[i:i+1, :] for v in self.inputs]
#             for j in range(num_samples):
#                 xj = [v[j:j+1, :] for v in self.inputs]
#                 for li, lgi in enumerate(self.loss_grads):
#                     if self.weights[li][i] == 0.: continue
#                     gi = np.concatenate([np.abs(wg).flatten() for wg in lgi(xi)])
#                     for lj, lgj in enumerate(self.loss_grads):
#                         if self.weights[lj][j] == 0.: continue
#                         gj = np.concatenate([np.abs(wg).flatten() for wg in lgj(xj)])
#                         gij = np.dot(gi, gj)
#                         ntk[li][lj][i, j] = gij
#         return ntk
#
#     def eval_diag_ntk(self):
#         # eval new gradients
#         num_samples = self.inputs[0].shape[0]
#         updated_NTK = [np.zeros((num_samples,)) for lgi in self.loss_grads]
#         # start_time = time()
#         for li, lgi in enumerate(self.loss_grads):
#             for i in range(num_samples):
#                 wi = self.weights[li][i]
#                 if wi == 0.: continue
#                 xi = [v[i:i + 1, :] for v in self.inputs]
#                 gi = np.concatenate([np.abs(wg).flatten() for wg in lgi(xi)])
#                 gij = np.dot(gi, gi)
#                 updated_NTK[li][i] = gij
#         # print(time() - start_time)
#         return updated_NTK
#
#     def eval_loss_weights(self, updated_grads):
#         # Method 1: normalization by PDE.
#         # eval max normalization on PDE.
#         ref_grad = []
#         for ws in updated_grads:
#             ref_grad.append(ws.sum())
#         normalization_grad = sum(ref_grad)
#         # evaluate new weights
#         new_weights = []
#         for i, type in enumerate(self.types):
#             ref_grad_i = ref_grad[i] if ref_grad[i] != 0. else np.mean(ref_grad)
#             new_weights.append(normalization_grad / ref_grad_i)
#         return new_weights
#
#     def update_loss_weights(self, epoch, updated_grads):
#         new_weights = self.eval_loss_weights(updated_grads)
#         norm = 1.0 #sum(new_weights)
#         self.loss_weights = []
#         # evaluate new weights
#         for i, wi in enumerate(self.model.loss_weights):
#             # gp_weight = new_weights[i] / K.get_value(wi)
#             # new_val = (1.0 - self.beta) * K.get_value(wi) + self.beta * gp_weight
#             new_val = new_weights[i]/norm
#             K.set_value(self.model.loss_weights[i], new_val)
#             self.loss_weights.append(new_val)
#         # print updates
#         print('\n+ adaptive_weights at epoch {}:'.format(epoch+1), self.loss_weights)
#
#     def update_loss_gradients(self, epoch, updated_grads):
#         self.loss_gradients = [np.linalg.norm(lgi) for lgi in updated_grads]
#
#     @staticmethod
#     def prepare_inputs(*args, **kwargs):
#         kwargs['method'] = 'NTK'
#         if len(args) == 1:
#             kwargs['freq'] = args[0]
#         elif len(args) > 0:
#             raise ValueError
#         return kwargs
#

@keras_export('keras.callbacks.NTKSampleWeight')
class NTKSampleWeight(Callback):
    """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
    """
    def __init__(self, model, data_generator,
                 beta=0.1, freq=100, log_freq=None,
                 hessian=False, types=None, **kwargs):
        super(NTKSampleWeight, self).__init__()
        # generate samples. 
        inputs, targets, weights = data_generator.get_data()
        # limit number of samples for performance concerns.
        self.inputs = inputs
        self.targets = targets
        self.weights = weights
        # eval loss and gradients.
        self.loss_grads = []
        for i in range(len(model.outputs)):
            # fixed batch size
            yp = model.outputs[i]
            ws = self.weights[i]
            gf = tf_gradients(yp, model.trainable_weights, unconnected_gradients='zero')
            self.loss_grads.append(K.function(model.inputs, gf))
        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.beta = beta
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        assert len(model.outputs) == len(types)

    @staticmethod
    def collect_weights(model):
        x_values = []
        for weights in model.trainable_weights:
            x_values.append(tf.reshape(weights, (-1,)))
        return tf.concat(x_values, axis=0)

    def update(self, epoch):
        loss_gradients = self.eval_diag_ntk()
        self.update_loss_weights(epoch, loss_gradients)
        self.update_loss_gradients(epoch, loss_gradients)

    def on_train_begin(self, logs={}):
        self.update(0)

    def on_epoch_begin(self, epoch, logs={}):
        if self.freq > 0 and (epoch + 1) % self.freq == 0:
            self.update(epoch)

    def on_epoch_end(self, epoch, logs={}):
        pass
        # # log gradient values
        # logs['adaptive_weights'] = self.loss_weights
        # # log norm2 of gradients
        # logs['loss_gradients'] = self.loss_gradients

    def eval_full_ntk(self):
        # eval new gradients
        num_samples = self.inputs[0].shape[0]
        ntk = [
            [np.zeros((num_samples,num_samples)) for lgi in self.loss_grads]
            for lgj in self.loss_grads
        ]
        for i in range(num_samples):
            xi = [v[i:i+1, :] for v in self.inputs]
            for j in range(num_samples):
                xj = [v[j:j+1, :] for v in self.inputs]
                for li, lgi in enumerate(self.loss_grads):
                    gi = np.concatenate([np.abs(wg).flatten() for wg in lgi(xi)])
                    for lj, lgj in enumerate(self.loss_grads):
                        gj = np.concatenate([np.abs(wg).flatten() for wg in lgj(xj)])
                        gij = np.dot(gi, gj)
                        ntk[li][lj][i, j] = gij
        return ntk

    def eval_diag_ntk(self):
        # eval new gradients
        num_samples = self.inputs[0].shape[0]
        updated_NTK = [np.zeros((num_samples,)) for lgi in self.loss_grads]
        for li, lgi in enumerate(self.loss_grads):
            for i in range(num_samples):
                xi = [v[i:i + 1, :] for v in self.inputs]
                gi = np.concatenate([np.abs(wg).flatten() for wg in lgi(xi)])
                gij = np.dot(gi, gi)
                updated_NTK[li][i] = gij
        return updated_NTK

    def eval_sample_weights(self, updated_grads):
        # Method 1: normalization by PDE.
        # evaluate new weights
        new_weights = []
        tr_ntk = np.sum([np.sum(wi) for wi in updated_grads])
        for i, ws in enumerate(updated_grads):
            ws[ws != 0.] = tr_ntk / ws[ws != 0.]
            new_weights.append(ws)
        return new_weights

    def update_loss_weights(self, epoch, updated_grads):
        new_weights = self.eval_sample_weights(updated_grads)
        self.sample_weights = []
        # evaluate new weights
        for i, wi in enumerate(self.model.sample_weights):
            # gp_weight = new_weights[i] / K.get_value(wi)
            # new_val = (1.0 - self.beta) * K.get_value(wi) + self.beta * gp_weight
            new_val = new_weights[i]
            self.weights[i][:] = new_val

    def update_loss_gradients(self, epoch, updated_grads):
        self.loss_gradients = [np.linalg.norm(lgi) for lgi in updated_grads]

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'NTK'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.AdaptiveSampleWeight')
class AdaptiveSampleWeight(Callback):
    """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
    """
    def __init__(self, model, data_generator,
                 beta=0.1, freq=100, log_freq=None,
                 hessian=False, types=None, **kwargs):
        super(AdaptiveSampleWeight, self).__init__()
        # generate samples. 
        inputs, targets, weights = data_generator.get_grid()
        # limit number of samples for performance concerns.
        self.inputs = inputs
        self.targets = targets
        self.weights = weights
        # eval loss and gradients.
        self.losses = []
        for i in range(len(model.outputs)):
            # fixed batch size
            yp = model.outputs[i]
            ys = self.targets[i]
            ws = self.weights[i]
            self.losses.append(
                K.function(model.inputs, yp)
            )
        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.beta = beta
        self.types = types
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq
        assert len(model.outputs) == len(types)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch + 1) % self.freq == 0:
            for i, loss in enumerate(self.losses):
                wi = np.abs(loss(self.inputs) - self.targets[i]).flatten()
                wi[wi != 0.] = 1. / wi[wi != 0.]
                wi *= (wi.size / np.sum(wi))
                self.weights[i][:] = (1-self.beta) * self.weights[i] + self.beta * wi

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'NTK'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


@keras_export('keras.callbacks.AdaptiveSampleWeight2')
class AdaptiveSampleWeight2(Callback):
    """ Callback that evaluate the adaptive weights based on the Gradient Pathologies approach by Wang et al.
    """
    def __init__(self, model, data_generator, loss=None,
                 beta=0.1, freq=100, log_freq=None,
                 hessian=False, types=None, **kwargs):
        super(AdaptiveSampleWeight2, self).__init__()
        # limit number of samples for performance concerns.
        self.data_generator = data_generator
        self.inputs = data_generator.get_grid()
        # eval loss and gradients.
        if loss is None:
            self.loss = K.function(model.inputs, model.outputs[0])
        else:
            self.loss = K.function(model.inputs, loss.outputs)
        self.freq = np.Inf if (freq==True or freq==0) else freq
        self.beta = beta
        if log_freq is None:
            self.log_freq = self.freq
        else:
            self.log_freq = log_freq

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch + 1) % self.freq == 0:
            loss = unpack_singleton(self.loss(self.inputs))
            self.data_generator.set_sample_weights(loss)

    @staticmethod
    def prepare_inputs(*args, **kwargs):
        kwargs['method'] = 'AdaptiveSampleWeight2'
        if len(args) == 1:
            kwargs['freq'] = args[0]
        elif len(args) > 0:
            raise ValueError
        return kwargs


def setup_adaptive_weight_callback(adaptive_weights, model, constraints, data_generator):
    """ Prepare adaptive_weight callback.

    # Arguments
        adaptive_weights:
        model:
        constraints:
        data_generator:
    :return:
    """
    if not isinstance(adaptive_weights, dict):
        adaptive_weights = NTKLossWeight.prepare_inputs(adaptive_weights)

    if adaptive_weights["method"].lower() in ("gp", "gradientpathology", "gradient_pathology"):
        return GradientPathologyLossWeight(
            model, data_generator=data_generator,
            types=[type(v).__name__ for v in constraints],
            **adaptive_weights
        )

    elif adaptive_weights["method"].lower() in ("gn", "gradnorm", "grad_norm"):
        return GradNormLossWeight(
            model, data_generator=data_generator,
            types=[type(v).__name__ for v in constraints],
            **adaptive_weights
        )

    elif adaptive_weights["method"].lower() in ("ntk", "neuraltangentkernel", "neural_tangent_kernel"):
        return NTKLossWeight(
            model, data_generator=data_generator,
            types=[type(v).__name__ for v in constraints],
            **adaptive_weights
        )

    elif adaptive_weights["method"].lower() in ("id", "inversedirichlet", "inverse_dirichlet"):
        return InverseDirichletLossWeight(
            model, data_generator=data_generator,
            types=[type(v).__name__ for v in constraints],
            **adaptive_weights
        )

    elif adaptive_weights["method"].upper() == "MTL":
        return MTLLossWeight(
            model, data_generator=data_generator,
            **adaptive_weights
        )

    elif adaptive_weights["method"].upper() == "SCORE":
        return ScoreLossWeight(
            model, data_generator=data_generator,
            **adaptive_weights
        )

    else:
        raise ValueError(
            'adaptive_weights method should be either of the followings: (GP, NTK)'
        )

