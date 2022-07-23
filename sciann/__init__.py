from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()

from . import constraints
from . import functionals
from . import models
from . import utils

from .functionals.functional import Functional
from .functionals.variable import Variable
from .functionals.field import Field
from .functionals.parameter import Parameter
from .models.model import SciModel
from .constraints import Constraint, PDE, Data, Tie

# Also importable from root
from .utils.math import *
from .utils import math
from .utils.utilities import reset_session, clear_session
from .utils.utilities import set_default_log_path, get_default_log_path
from .utils.utilities import set_random_seed
from .utils.utilities import set_floatx
from .utils.utilities import get_bibliography

import json
from urllib import request
from pkg_resources import parse_version


# SciANN.
__author__ = "Ehsan Haghighat"
__email__ = "ehsanh@mit.edu"
__copyright__ = "Copyright 2019, Physics-Informed Deep Learning"
__credits__ = []
__url__ = "http://github.com/sciann/sciann]"
__license__ = "MIT"
__version__ = "0.6.7.1"
__cite__ = \
    '@article{haghighat2021sciann, \n' +\
    '    title={SciANN: A Keras/TensorFlow wrapper for scientific computations and physics-informed deep learning using artificial neural networks}, \n' +\
    '    author={Haghighat, Ehsan and Juanes, Ruben}, \n' +\
    '    journal={Computer Methods in Applied Mechanics and Engineering}, \n' +\
    '    year={2021}, \n' +\
    '    url = {https://doi.org/10.1016/j.cma.2020.113552}, \n' +\
    '    howpublished={https://github.com/sciann/sciann.git}, \n' +\
    '}'

# Import message.
_header = '---------------------- {} {} ----------------------'.format(str.upper(__name__), str(__version__))
_footer = len(_header)*'-'
__welcome__ = \
    '{} \n'.format(_header) +\
    'For details, check out our review paper and the documentation at: \n' +\
    ' +  "https://www.sciencedirect.com/science/article/pii/S0045782520307374", \n' +\
    ' +  "https://arxiv.org/abs/2005.08803", \n' +\
    ' +  "https://www.sciann.com". \n' +\
    '\n ' +\
    'Need support or would like to contribute, please join sciann`s slack group: \n' +\
    ' +  "https://join.slack.com/t/sciann/shared_invite/zt-ne1f5jlx-k_dY8RGo3ZreDXwz0f~CeA" ' +\
    '\n \n' +\
    'TensorFlow Version: {} \n'.format(tf.__version__) +\
    'Python Version: {} \n'.format(sys.version)
    # '{} \n'.format(__cite__) +\
    # _footer


import os
if 'SCIANN_WELCOME_MSG' in os.environ.keys() and \
        os.environ['SCIANN_WELCOME_MSG']=='-1':
    pass
else:
    print(__welcome__)

# set default logging directory.
set_default_log_path(os.path.join(os.getcwd(), "logs"))
initialize_bib(os.path.join(os.path.dirname(__file__), 'references', 'bibliography'))

# check sciann version.
try:
    url = 'https://pypi.python.org/pypi/sciann/json'
    releases = json.loads(request.urlopen(url, timeout=1).read())['releases']
    releases = sorted(releases, key=parse_version, reverse=True)
    if releases.index(__version__) > 0:
        print(f'Outdated SciANN installation is found (V-{__version__}). '
              f'Get the latest version (V-{releases[0]}):  \n '
              f'     > pip [--user] install --upgrade sciann  ')
except:
    pass
