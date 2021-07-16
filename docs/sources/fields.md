# Intro

`Field` is a layer to define outputs of each Functional. It is very much similar to Keras' `Dense` layer. 

It is not necessary to be defined explicitly, however, if you are expecting multiple outputs, it is better to be defined using `Field`.  

```python
from sciann import Field

Fx = Field(name='Fx', units=10)
```

---

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


    
