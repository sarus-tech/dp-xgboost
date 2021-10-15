#! /bin/bash
PYTHON=python3

cd ../build/
# only build the xgboost target
make xgboost -j4

# install latest lib
cd ../python-package
$PYTHON -m pip install -e .
