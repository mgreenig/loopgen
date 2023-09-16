#!/bin/bash

env_name="loopgen"

# check OS
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     OS=Linux;;
    Darwin*)    OS=Mac;;
    *)          OS="Other"
esac

# set up conda environment
conda create -n $env_name
eval "$(conda shell.bash hook)"
conda activate $env_name

# if on a mac and not on an ARM chip change the env config
if [ $OS == "Mac" ] && [ $(uname -p) != "arm" ]; then
  conda config --env --set subdir osx-64
fi

conda install nomkl

mamba env update -n $env_name -f envs/environment.yml

# make sure to install compatible torch packages if on Linux to allow GPU usage
if [ $OS == "Linux" ]; then
  pip install https://download.pytorch.org/whl/cu117/torch-2.0.1%2Bcu117-cp311-cp311-linux_x86_64.whl
  pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.1%2Bpt20cu117-cp311-cp311-linux_x86_64.whl
  pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_sparse-0.6.17%2Bpt20cu117-cp311-cp311-linux_x86_64.whl
  pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_cluster-1.6.1%2Bpt20cu117-cp311-cp311-linux_x86_64.whl
elif [ $OS == "Mac" ]; then
  pip install https://download.pytorch.org/whl/cpu/torch-2.0.1-cp311-none-macosx_10_9_x86_64.whl
  pip install numpy==1.25.2 --force-reinstall
  pip install scipy==1.11.1 --force-reinstall
  pip install https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_sparse-0.6.17-cp311-cp311-macosx_10_9_universal2.whl
  pip install https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_scatter-2.1.1-cp311-cp311-macosx_10_9_universal2.whl
  pip install https://data.pyg.org/whl/torch-2.0.0%2Bcpu/torch_cluster-1.6.1-cp311-cp311-macosx_10_9_universal2.whl
else
  pip install torch==2.0.1
  pip install torch-sparse==0.6.17
  pip install torch-scatter==2.1.1
  pip install torch-cluster==1.6.1
fi

# install pip requirements
pip install -r envs/pip_requirements.txt

# install the package itself
pip install .

# set lib path for C++ libraries
env_path=$(conda info --base)/envs/$env_name

activate_env_vars=$env_path/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=$env_path/lib:$LD_LIBRARY_PATH" > $activate_env_vars

deactivate_env_vars=$env_path/etc/conda/deactivate.d/env_vars.sh
echo "unset LD_LIBRARY_PATH" > $deactivate_env_vars