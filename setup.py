from setuptools import setup
from setuptools import find_packages

long_description = '''
SciANN is an Artificial Neural Netowork library, 
based on Python, Keras, and TensorFlow, designed
to perform scientific computations, solving ODEs 
and PDEs, curve-fitting, etc, very efficiently.

Read the documentation at: https://sciann.com

SciANN is compatible with Python 2.7-3.6
and is distributed under the MIT license.
'''

setup(
    name='SciANN',
    version='0.6.7.6',
    description='A Keras/Tensorflow wrapper for scientific computations and physics-informed deep learning using artificial neural networks.',
    long_description=long_description,
    author='Ehsan Haghighat',
    author_email='ehsan@sciann.com',
    license='MIT',
    url='https://github.com/sciann/sciann',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'pyyaml',
                      'h5py',
                      'sklearn',
                      'pybtex',
                      'tensorflow>=2.6.0,<=2.9.5',
                      ],
    extras_require={
          'visualize': ['pydot>=1.2.4', 'matplotlib'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'pandas',
                    'requests',
                    'graphviz',
                    'pydot',
                    'markdown',
                    'matplotlib',
                    'pyux',
                    ],
    },
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
    packages=find_packages(),
    include_package_data=True
)
