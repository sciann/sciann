
export ENV_NAME=tf29
export TF_VER=2.9.0
export PY_VER=3.9

# source ~/.miniconda_zshrc

conda create -n $ENV_NAME python=$PY_VER -y
conda activate $ENV_NAME
conda install -c apple tensorflow-deps=$TF_VER -y
pip install tensorflow-macos==$TF_VER

# pip install tensorflow-metal

conda install -c conda-forge -y pandas jupyterlab scipy scikit-learn matplotlib 
pip install plotly
pip install -e .

cd examples
python example-fitting-1d.py 
