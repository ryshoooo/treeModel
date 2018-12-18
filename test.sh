#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

echo "Creating conda virtual testing environment"
conda create -y --no-default-packages -n test_treemodel python=3.6
source activate test_treemodel

pip install $DIR
pip install -r $DIR/test_requirements.txt

python -m unittest discover $DIR/test

source deactivate
conda remove -y -n test_treemodel --all