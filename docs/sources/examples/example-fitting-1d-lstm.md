
# Curve fitting in 1D

Here, a 1D curve fitting example is explored. Imagine, a synthetic data
generated from \\( \sin(x) \\) over the range of \\( [0, 2\pi] \\).

To train a neural network model on this curve, you should first define a `Variable`.

A neural network with three layers, each containing 10 neurons, and with `tanh` activation function is then generated
using the `Functional` class.

The target is imposed on the output using the `Data` class from `Constraint`, and passed to the `SciModel` to form a
Sciann model.


```python
# Curve fitting in 1D

Here, a 1D curve fitting example is explored. Imagine, a synthetic data
generated from \\\( \sin(x) \\\) over the range of \\\( [0, 2\pi] \\\).

To train a neural network model on this curve, you should first define a `Variable`.

A neural network with three layers, each containing 10 neurons, and with `tanh` activation function is then generated
using the `Functional` class.

The target is imposed on the output using the `Data` class from `Constraint`, and passed to the `SciModel` to form a
Sciann model.
'''

import numpy as np
# from sciann.functionals.rnn_variable import RNNVariable
# from sciann.functionals.rnn_functional import RNNFunctional
# from sciann.functionals.rnn_field import RNNField
# from sciann import SciModel
# from sciann.utils import diff, set_random_seed
# from sciann.constraints import Data, Tie
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

# set_random_seed(1234)

tunits = 3

# Synthetic data generated from sin function over [0, 2pi]
x_true = np.linspace(0, np.pi*2, 100)
y_true = np.sin(x_true)

dataset = timeseries_dataset_from_array(y_true[:-3], y_true[3:], sequence_length=tunits, batch_size=10)
for batch in dataset:
    inputs, targets = batch
    print('ins: {}  outs: {}'.format(inputs,targets))


raise ValueError


# The network inputs should be defined with Variable.
t = RNNVariable(tunits, name='t', dtype='float64')

# Each network is defined by Functional.
y = RNNFunctional(
    'y', t, [5],
    activation='tanh',
    recurrent_activation='linear',
    kernel_initializer=1.0,
    recurrent_initializer=1.0,
    bias_initializer=1.0,
    rnn_type="SimpleRNN"
)
dy= diff(y, t)

def rnn_cell(x):
    y = []
    actf = lambda x: x
    for i, xi in enumerate(x):
        if i==0:
            y.append(
                actf(np.ones([1,5]) + xi*np.ones([1,5]))
            )
        else:
            y.append(
                actf(np.ones([1, 5]) + xi * np.ones([1, 5]) + np.matmul(y[-1], np.ones([5, 5])))
            )

    actf = lambda x: np.tanh(x)
    y = [actf(yi) for yi in y]
    return y

def rnn_net(x):
    actf = lambda x: x
    ys = rnn_cell(x)
    return np.concatenate([1 + np.matmul(actf(y), np.ones([5,1])) for y in ys], axis=-1).flatten()

test_rnn = lambda i: y.eval(x_true[i,:]) - rnn_net(x_true[i, :])

for i in range(10):
    print(test_rnn(i))

raise ValueError

# Define the target (output) of your model.
c1 = Data(y)
# c2 = Data((y[1]-y[0])/dx)
c2 = Data(dy)


# The model is formed with input `x` and condition `c1`.
model = SciModel(t, [c1, c2])

# Training: .train runs the optimization and finds the parameters.
model.train(
    x_true.reshape(-1, tunits, 1), 
    [y_true.reshape(-1, tunits, 1), dy_true.reshape(-1, tunits, 1)], 
    batch_size=1000000, epochs=20000, learning_rate=0.01
)

# used to evaluate the model after the training.
# x_pred = np.linspace(0, np.pi*4, 20000).reshape(-1, tunits, 1)
x_pred = np.linspace(0, np.pi*4, 200).reshape(-1, 1)
dx = np.diff(x_pred.flatten()).mean()
x_pred = x_pred + np.linspace(0, dx*(tunits-1), tunits).reshape(1, -1)

y_pred = y.eval(model, x_pred)
# dy_pred = dy.eval(model, x_pred)

y_star = np.sin(x_pred)
dy_star = np.cos(x_pred)

import matplotlib.pyplot as plt
plt.plot(x_pred[:,0], y_pred[:,0], x_pred[:,0], y_star[:,0])
# plt.plot(x_pred, dy_pred.reshape(-1), x_pred, dy_star)
plt.show()
```