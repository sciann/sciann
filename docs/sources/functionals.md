# Intro

A combination of neural network layers form a `Functional`. 

Mathematically, a `functional` is a general mapping from input set \\(X\\) onto some output set \\(Y\\). Once the parameters of this transformation are found, this mapping is called a `function`. 

`Functional`s are needed to form `SciModels`. 

A `Functional` is a class to form complex architectures (mappings) from inputs (`Variables`) to the outputs. 


```python
from sciann import Variable, Functional

x = Variable('x')
y = Variable('y')

Fxy = Functional('Fxy', [x, y], 
                 hidden_layers=[10, 20, 10],
                 activation='tanh')
```

`Functionals` can be plotted when a `SciModel` is formed. A minimum of one `Constraint` is needed to form the SciModel

```python
from sciann.constraints import Data
from sciann import SciModel

model = SciModel(x, Data(Fxy), 
                 plot_to_file='output.png')
```

---

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/mlp_functional.py#L31)</span>
### MLPFunctional

```python
sciann.functionals.mlp_functional.MLPFunctional(inputs, outputs, layers)
```

Configures the Functional object (Neural Network).

__Arguments__

- __fields__: String or Field.
    [Sub-]Network outputs.
    It can be of type `String` - Associated fields will be created internally.
    It can be of type `Field` or `Functional`
- __variables__: Variable.
    [Sub-]Network inputs.
    It can be of type `Variable` or other Functional objects.
- __hidden_layers__: A list indicating neurons in the hidden layers.
    e.g. [10, 100, 20] is a for hidden layers with 10, 100, 20, respectively.
- __activation__: defaulted to "tanh".
    Activation function for the hidden layers.
    Last layer will have a linear output.
- __output_activation__: defaulted to "linear".
    Activation function to be applied to the network output.
- __res_net__: (True, False). Constructs a resnet architecture.
    Defaulted to False.
- __kernel_initializer__: Initializer of the `Kernel`, from `k.initializers`.
- __bias_initializer__: Initializer of the `Bias`, from `k.initializers`.
- __kernel_regularizer__: Regularizer for the kernel.
    To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
- __bias_regularizer__: Regularizer for the bias.
    To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').
    Note: Only network inputs should be set.
- __trainable__: Boolean.
    False if network is not trainable, True otherwise.
    Default value is True.

__Raises__

- __ValueError__:
- __TypeError__:
    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/variable.py#L10)</span>
### Variable

```python
sciann.functionals.variable.Variable(name=None, units=1, tensor=None, dtype=None)
```

Configures the `Variable` object for the network's input.

__Arguments__

- __name__: String.
    Required as derivatives work only with layer names.
- __units__: Int.
    Number of feature of input var.
- __tensor__: Tensorflow `Tensor`.
    Can be pass as the input path.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').

__Raises__


    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/field.py#L14)</span>
### Field

```python
sciann.functionals.field.Field(name=None, units=1, activation=<function linear at 0x7f87707bc8b0>, kernel_initializer=None, bias_initializer=None, kernel_regularizer=None, bias_regularizer=None, trainable=True, use_bias=True, dtype=None)
```

Configures the `Field` class for the model outputs.

__Arguments__

- __name__: String.
    Assigns a layer name for the output.
- __units__: Positive integer.
    Dimension of the output of the network.
- __activation__: Callable.
    A callable object for the activation.
- __kernel_initializer__: Initializer for the kernel.
    Defaulted to a normal distribution.
- __bias_initializer__: Initializer for the bias.
    Defaulted to a normal distribution.
- __kernel_regularizer__: Regularizer for the kernel.
    To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
- __bias_regularizer__: Regularizer for the bias.
    To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
- __trainable__: Boolean to activate parameters of the network.
- __use_bias__: Boolean to add bias to the network.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').

__Raises__


    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/parameter.py#L31)</span>
### Parameter

```python
sciann.functionals.parameter.Parameter(val=1.0, min_max=None, inputs=None, name=None, non_neg=None)
```

Parameter functional to be used for parameter inversion.
Inherited from Dense layer.

__Arguments__

- __val__: float.
    Initial value for the parameter.
- __min_max__: [MIN, MAX].
    A range to constrain the value of parameter.
    This constraint will overwrite non_neg constraint if both are chosen.
- __inputs__: Variables.
    List of `Variable`s to the parameters.
- __name__: str.
    A name for the Parameter layer.
- __non_neg__: boolean.
    True (default) if only non-negative values are expected.
- __**kwargs__: keras.layer.Dense accepted arguments.

    
----

### eval


```python
eval()
```


Evaluates the functional object for a given input.

__Arguments__

(SciModel, Xs): 
Evalutes the functional object from the beginning 
of the graph defined with SciModel. 
The Xs should match those of SciModel. 

(Xs):
Evaluates the functional object from inputs of the functional. 
Xs should match those of inputs to the functional. 

__Returns__

Numpy array of dimensions of network outputs. 

__Raises__

- __ValueError__:
- __TypeError__:
    
----

### get_weights


```python
get_weights(at_layer=None)
```


Get the weights and biases of different layers.

__Arguments__

- __at_layer__: 
    Get the weights of a specific layer. 

__Returns__

List of numpy array. 
    
----

### set_weights


```python
set_weights(weights)
```


Set the weights and biases of different layers.

__Arguments__

- __weights__: Should take the dimensions as the output of ".get_weights"

__Returns __

    
----

### reinitialize_weights


```python
reinitialize_weights()
```


Re-initialize the weights and biases of a functional object.

__Arguments__


__Returns __


    
----

### count_params


```python
count_params()
```


Total number of parameters of a functional.

__Arguments__


__Returns __

Total number of parameters.
    
----

### set_trainable


```python
set_trainable(val, layers=None)
```


Set the weights and biases of a functional object trainable or not-trainable.
Note: The SciModel should be called after this.

__Arguments__

- __val__: (Ture, False)
- __layers__: list of layers to be set trainable or non-trainable.
    defaulted to None.

__Returns __

    
----

### split


```python
split()
```


In the case of `Functional` with multiple outputs,
you can split the outputs and get an associated functional.

__Returns__

(f1, f2, ...): Tuple of splitted `Functional` objects
    associated to each output.
    
