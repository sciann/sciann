'''
# Curve fitting in 1D

Here, a 1D curve fitting example is explored. Imagine, a synthetic data
generated from \\\( \sin(x) \\\) over the range of \\\( [0, 2\pi] \\\).

To train a neural network model on this curve, you should first define a `Variable`.

A neural network with three layers, each containing 10 neurons, and with `tanh` activation function is then generated
using the `Functional` class.

The target is imposed on the output using the `Data` class from `Constraint`, and passed to the `SciModel` to form a
SciANN model.
'''

import numpy as np
from sciann import Variable, Functional, SciModel, Parameter
from sciann.constraints import Data, MinMax
import sciann as sn 
import time


# Synthetic data generated from sin function over [0, 2pi]
x_true = np.linspace(0, np.pi*2, 10000)
y_true = np.sin(x_true)
dy_true = np.cos(x_true)

# The network inputs should be defined with Variable.
x = Variable('x')
xf = Functional('xf', x)
xf.set_trainable(False)

# Each network is defined by Functional.
y = Functional('y', x, [10, 10, 10], activation=['tanh'])
dy_dx = sn.diff(y, x)

d = Parameter(2.0, inputs=x)

# Define the target (output) of your model.
c1 = Data(y)

# The model is formed with input `x` and condition `c1`.
model = SciModel(x, [y, dy_dx], optimizer='adam')
model.summary()

start_time = time.time()

# Training: .train runs the optimization and finds the parameters.
model.train(x_true,
            [y_true, dy_true],
            epochs=100,
            learning_rate={"scheduler": "ExponentialDecay",
                           "initial_learning_rate": 1e-3,
                           "final_learning_rate": 1e-5,
                           "decay_epochs": 10,
                           "verify": False},
            batch_size=32,
            adaptive_weights={'method': "SA", "eta": 0.01}
            )

print(f"Training finished in {time.time()-start_time}s. ")

# used to evaluate the model after the training.
y_pred = y.eval(model, x_true)

# print(x_true.shape, y_pred.shape)
import matplotlib.pyplot as plt
plt.plot(x_true, y_true, x_true, y_pred)
plt.show()