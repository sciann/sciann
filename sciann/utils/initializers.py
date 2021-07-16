
from tensorflow.python import keras as k


class SciKernelInitializer(k.initializers.VarianceScaling):

    def __init__(self, lay=0, seed=None):
        self.lay = lay

        self.w0 = 1.0
        scale = 1.0
        distribution = 'truncated_normal'
        if lay==0:
            mode = 'fan_in'
        else:
            mode = 'fan_avg'

        # # based on Wang Yang's work
        # if lay==0:
        #     self.w0 = lambda x: k.sqrt(x.shape[-1])
        # else:
        #     self.w0 = lambda x: 1./k.sqrt(x.shape[-1])

        super(SciKernelInitializer, self).__init__(
            scale=scale,
            mode=mode,
            distribution=distribution,
            seed=seed
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
            'lay': self.lay,
            'bias': self.bias
        }
        return dict(list(base_config.items()) + list(config.items()))


class SciBiasInitializer(k.initializers.RandomUniform):

    def __init__(self, lay=0, seed=None):
        self.w0 = 1.0
        self.lay = lay

        super(SciBiasInitializer, self).__init__(seed=seed)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
            'lay': self.lay
        }
        return dict(list(base_config.items()) + list(config.items()))


class SciOrthogonalInitializer(k.initializers.Orthogonal):

    def __init__(self, actf='linear', lay=0, bias=False, seed=None):
        self.actf = actf
        self.bias = bias

        if actf in ('linear', 'relu'):
            self.w0 = 1.0
            scale = 2.0
        elif actf in ('tanh', 'atan'):
            self.w0 = 1.0
            scale = 1.0
        elif actf in ('sin', 'cos'):
            self.w0 = 30.0 if lay==0 else 1.0
            scale = 2.0
        else:
            self.w0 = 1.0
            scale = 1.0

        if bias is True:
            self.w0 = 1./self.w0

        super(SciOrthogonalInitializer, self).__init__(
            gain=scale,
            seed=seed
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            'w0': self.w0,
            'actf': self.actf,
            'bias': self.bias
        }
        return dict(list(base_config.items()) + list(config.items()))


k.utils.generic_utils.get_custom_objects().update({
    'SciKernelInitializer': SciKernelInitializer,
    'SciOrthogonalInitializer': SciOrthogonalInitializer
})
