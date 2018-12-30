#!/usr/bin/env bash
# This simple script creates sphinx documentation for the treemodel library.
# The HTML files can be found afterwards in the directory `_docs_build`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

echo "Creating conda virtual documentation environment"
conda create -y --no-default-packages -n test_treemodel python=3.6
source activate test_treemodel

pip install $DIR
pip install -r $DIR/docs_requirements.txt

sphinx-apidoc --implicit-namespaces -o $DIR/docs -f $DIR/treemodel
sphinx-build -b html $DIR/docs $DIR/_docs_build

source deactivate
conda remove -y -n test_treemodel --all