#!/bin/bash

# wget http://repo.continuum.io/miniconda/Miniconda3-3.7.3-Linux-x86_64.sh
# bash Miniconda3-3.7.3-Linux-x86_64.sh
# conda config --add channels http://conda.binstar.org/bjodah
# conda install numpy=1.9.0 scipy=0.14.0 matplotlib=1.4.0 cython=0.21 mako periodictable quantities pytest future pip sphinx numpydoc pycompilation pycodeexport sympy
# Rename libm.so* in $CONDA_ROOT/pkgs/system-5.8-1/lib/

CONDAROOT=$(conda info -s | grep sys.prefix | awk '{print $2}')
PYTHONPATH=$CONDAROOT/lib/python3.4/site-packages:/home/asv/miniconda3/lib/python3.4/site-packages/Mako-1.0.0-py3.4.egg asv run $@
