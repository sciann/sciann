
# Curve fitting in 1D

Here, a 1D curve fitting example is explored. Imagine, a synthetic data
generated from \\( \sin(x) \\) over the range of \\( [0, 2\pi] \\).

To train a neural network model on this curve, you should first define a `Variable`.

A neural network with three layers, each containing 10 neurons, and with `tanh` activation function is then generated
using the `Functional` class.

The target is imposed on the output using the `Data` class from `Constraint`, and passed to the `SciModel` to form a
SciANN model.


```python
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


# Synthetic data generated from sin function over [0, 2pi]
x_true = np.linspace(0, np.pi*2, 10000)
y_true = np.sin(x_true)

# The network inputs should be defined with Variable.
x = Variable('x', dtype='float64')
xf = Functional('xf', x)
xf.set_trainable(False)

# Each network is defined by Functional.
y = Functional('y', xf, [10, 10, 10], activation='tanh')

d = Parameter(2.0, inputs=x)

# Define the target (output) of your model.
c1 = Data(y)

# The model is formed with input `x` and condition `c1`.
model = SciModel(x, [c1], optimizer='adam')
model.summary()

# Training: .train runs the optimization and finds the parameters.
model.train(x_true,
            [y_true],
            learning_rate={"scheduler":"SineExponentialDecay", "verify": True},
            batch_size=32,
            epochs=2,
            adaptive_weights={'method': "NTK", 'freq': 10, "use_score": True, "alpha": 1.}
            )

# used to evaluate the model after the training.
y_pred = y.eval(model, x_true)

print(x_true.shape, y_pred.shape)```