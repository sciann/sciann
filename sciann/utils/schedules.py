""" schedules module define LR schedulers for training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.callbacks import LearningRateScheduler as LearningRateSchedule
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def sci_exponential_decay(initial_learning_rate=1e-3,
                          final_learning_rate=1e-5,
                          decay_epochs=1000,
                          delay_epochs=0,
                          verify=False,
                          scheduler="exponential_decay"):
    """Applies exponential decay to the learning rate.
    Example:
    ``` python
        sci_exponential_decay(1e-3, 1e-5, 10000)
    ```
    Args:
        initial_learning_rate: The initial learning rate.
            Defaults to 1e-3.
        final_learning_rate: The final learning rate.
            Defaults to 1e-5.
        decay_epochs: number of epochs, over which the exponential decay is applied.
            Defaults to 1000.
        delay_epochs: number of epochs, over which the initial learning rate is used.
            Defaults to 0 (No delay).
        verify: Boolean. Plots the learning rate schedule.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
    """
    initial_learning_rate = lr0 = initial_learning_rate
    final_learning_rate = lr1 = final_learning_rate
    decay_epochs = decay_epochs
    decay_rate = np.log(lr1 / lr0) / (decay_epochs)
    epoch_vals = np.linspace(0, decay_epochs, 201)
    lr_vals = lr0 * np.exp(decay_rate * epoch_vals)
    
    # account for delay. 
    if delay_epochs > 0:
        epoch_vals = np.concatenate([[0, delay_epochs], epoch_vals+delay_epochs])
        lr_vals = np.concatenate([[initial_learning_rate, initial_learning_rate], lr_vals])

    if verify is True:
        plot_lr_schedule(epoch_vals, lr_vals)

    func = interp1d(epoch_vals, lr_vals,
                    bounds_error=False,
                    fill_value=(initial_learning_rate, final_learning_rate))

    return LearningRateSchedule(lambda i: float(func(i)))


def sci_sinusoidal_exponential_decay(
        initial_learning_rate=1e-3,
        final_learning_rate=1e-5,
        decay_epochs=1000,
        delay_epochs=0,
        sine_freq=10,
        sine_decay_rate=0.5,
        verify=False,
        scheduler="sinusoidal_exponential_decay"):
    """Applies exponential decay to the learning rate.
    Args:
        initial_learning_rate: The initial learning rate.
            Defaults to 1e-3.
        final_learning_rate: The final learning rate.
            Defaults to 1e-5.
        decay_epochs: number of epochs, over which the exponential decay is applied.
            Defaults to 1000.
        delay_epochs: number of epochs, over which the initial learning rate is used.
            Defaults to 0 (No delay).
        sine_freq: number of sinusoidal oscillations in learning rate.
            defaults to 10.
        sine_decay_rate: exponential decay on the amplitude of sin wave.
            defaults to 0.5.
        verify: Boolean. Plots the learning rate schedule.
    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
    """
    lr0 = initial_learning_rate
    lr1 = final_learning_rate
    decay_rate = np.log(lr1 / lr0) / (decay_epochs)

    th = np.arctan(decay_rate)

    epoch_vals = np.linspace(0, decay_epochs, 201)
    base_lr_vals = lr0 * np.exp(decay_rate * epoch_vals)
    decay_axis = epoch_vals * np.cos(th) + np.log(base_lr_vals / lr0) * np.sin(th)
    normalized_epoch_vals = 2 * np.pi * epoch_vals / epoch_vals.max()
    sin_lr_vals = np.exp(-sine_decay_rate * normalized_epoch_vals) * \
                  np.sin(-sine_freq * normalized_epoch_vals)
    lr_vals = lr0 * np.exp(decay_axis * np.sin(th) + sin_lr_vals * np.cos(th))

    # account for delay. 
    if delay_epochs > 0:
        epoch_vals = np.concatenate([[0, delay_epochs], epoch_vals+delay_epochs])
        lr_vals = np.concatenate([[initial_learning_rate, initial_learning_rate], lr_vals])

    if verify is True:
        plot_lr_schedule(epoch_vals, lr_vals)

    func = interp1d(epoch_vals, lr_vals,
                    bounds_error=False,
                    fill_value=(initial_learning_rate, final_learning_rate))
    return LearningRateSchedule(lambda i: float(func(i)))


def sci_learning_rate_schedule(
        lr_epochs=None,
        lr_values=None,
        verify=False,
        scheduler="learning_rate_schedule"):
    """uses discrete values to schedule the learning rate.
    Args:
        lr_epochs: list or numpy array of epochs at which learning rate values are given.
        lr_values: List or numpy array of learning rates.
        'sci_learning_rate_schedule'.
        verify: Boolean. Plots the learning rate schedule.
    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
    """
    func = interp1d(lr_epochs, lr_values,
                    bounds_error=False,
                    fill_value=(lr_values[0], lr_values[-1]))

    if verify is True:
        plot_lr_schedule(lr_epochs, lr_values)

    return LearningRateSchedule(lambda i: float(func(i)))


def setup_lr_scheduler(learning_rate_scheduler):
    assert isinstance(learning_rate_scheduler, dict), \
        'Expecting a dictionary describing the scheduler. '
    if 'scheduler' in learning_rate_scheduler.keys():
        if learning_rate_scheduler['scheduler'].lower() in \
                alternative_names(["exponential", "decay"]):
            try:
                return sci_exponential_decay(**learning_rate_scheduler)
            except:
                help(sci_exponential_decay)
                raise ValueError('Inconsistent inputs for `sci_exponential_decay`.')
        elif learning_rate_scheduler['scheduler'].lower() in \
                alternative_names(["sine", "exponential", "decay"]):
            try:
                return sci_sinusoidal_exponential_decay(**learning_rate_scheduler)
            except:
                help(sci_sinusoidal_exponential_decay)
                raise ValueError('Inconsistent inputs for `sci_sinusoidal_exponential_decay`.')
        elif learning_rate_scheduler['scheduler'].lower() in \
                alternative_names(["learning", "rate", "scheduler"]):
            try:
                return sci_learning_rate_schedule(**learning_rate_scheduler)
            except:
                help(sci_learning_rate_schedule)
                raise ValueError('Inconsistent inputs for `sci_learning_rate_schedule`.')
        elif learning_rate_scheduler['scheduler'].lower() in ('default',):
            try:
                return ReduceLROnPlateau(
                    monitor='loss', factor=0.5,
                    patience=learning_rate_scheduler['reduce_lr_after'],
                    verbose=1, mode='auto',
                    min_delta=learning_rate_scheduler['reduce_lr_min_delta'],
                    min_lr=0.
                )
            except:
                raise ValueError(
                    'Incorrect values for `reduce_lr_after` or `reduce_lr_min_delta`. '
                )

        else:
            raise ValueError('Unrecognized scheduler.')
    else:
        raise ValueError('`scheduler` not found in the dictionary. ')


def alternative_names(name):
    assert isinstance(name, list)
    assert all([isinstance(v, str) for v in name])
    alts = [
        "".join(name).lower(),
        "_".join(name).lower(),
        "".join([v[0] for v in name]).lower(),
    ]
    return alts


def plot_lr_schedule(epochs, lrs):
    """ Plotting (epochs, lr) for debugging purposes.
    Args:
        epochs: list or numpy array of epochs at which learning rate values are given.
        lrs: List or numpy array of learning rates.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, lrs)
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("LR Schedule")

    ax[1].semilogy(epochs, lrs)
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("LR Schedule")
    plt.subplots_adjust(0.15, 0.15, 0.85, 0.85, 0.3, 0.2)

    plt.show()
    plt.close(fig)
