

```python
import matplotlib.pyplot as plt
import sciann as sn


omega = 2*np.pi/2.0
omega_bar = 2*np.pi/1.5
A = 1.0
beta = 1.0

tunits = 2
NDATA = 1000

t_data = np.linspace(0, 4*np.pi, NDATA).reshape(-1,1)
dt = np.diff(t_data.flatten()).mean()
t_data = t_data + np.linspace(0, dt*(tunits-1), tunits).reshape(1, -1)

y_data = A*(np.sin(omega*t_data) - beta*np.sin(omega_bar*t_data))

# Add noise
# y_noise = 0.15*np.std(y_data)*np.random.randn(NDATA)


t = sn.functionals.RNNVariable(tunits, 't')
y = sn.functionals.RNNFunctional('y', t, [1], 'sin', recurrent_activation='sin')

mRNN = sn.SciModel(t, y)

mRNN.train(t_data, y_data, learning_rate=0.001, epochs=10000, batch_size=100)

```