'''
# Curve fitting in 1D with Fourier features

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
from sciann.utils.math import diff
import sciann as sn

sn.set_random_seed(1234)
# Synthetic data generated from sin function over [0, 2pi]
x_true = np.linspace(0, np.pi*2, 10000)
y_true = np.sin(x_true)

# The network inputs should be defined with Variable.
x = Variable('x', dtype='float64')
xf = sn.fourier(x, 10)

# Each network is defined by Functional.
y1 = sn.Field('y1', 10)
y2 = sn.Field('y2', 10)
y1, y2 = sn.Functional([y1,y2], xf, [10, 10, 10], 'l-tanh', output_activation='tanh')

y = sn.Functional('y', [xf*y1, xf*y2])

d = Parameter(10.0, inputs=x, name='d')

# Define the target (output) of your model.
c1 = Data(y)

L = d*diff(y, x, order=2) + y

# The model is formed with input `x` and condition `c1`.
model = SciModel(x, [c1, sn.PDE(L)])

# Tra: .train runs the optimization and finds the parameters.
history = model.train(
    x_true,
    [y_true, 'zeros'],
    batch_size=32,
    epochs=100,
    adaptive_weights={"method": "NTK", "freq": 10},
    log_parameters=[d]
)

# used to evaluate the model after the training.
y_pred = y.eval(model, x_true)
d_pred = d.eval(model, x_true)

sn.get_bibliography(format="bibtex")  #bibtexml
