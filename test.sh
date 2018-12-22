#!/usr/bin/env bash

# This simple script runs unittests on all Python major versions 3.4 and higher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
OLD_PYTHONPATH=$PYTHONPATH
PYTHONPATH=$OLD_PYTHONPATH:$DIR
export PYTHONPATH

echo "=========== PYTHON 3.6 ============"
conda create -y --no-default-packages -n test_treemodel python=3.6
source activate test_treemodel

pip install $DIR
pip install -r $DIR/test_requirements.txt

python -m unittest discover $DIR/tests

source deactivate
conda remove -y -n test_treemodel --all

echo "=========== PYTHON 3.5 ============"
conda create -y --no-default-packages -n test_treemodel python=3.5
source activate test_treemodel

pip install $DIR
pip install -r $DIR/test_requirements.txt

python -m unittest discover $DIR/tests

source deactivate
conda remove -y -n test_treemodel --all

echo "=========== PYTHON 3.4 ============"
conda create -y --no-default-packages -n test_treemodel python=3.4
source activate test_treemodel

pip install $DIR
pip install -r $DIR/test_requirements.txt

python -m unittest discover $DIR/tests

source deactivate
conda remove -y -n test_treemodel --all

PYTHONPATH=$OLD_PYTHONPATH