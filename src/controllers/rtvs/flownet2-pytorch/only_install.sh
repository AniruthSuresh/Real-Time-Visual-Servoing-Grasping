#!/bin/bash
set -e
cd ./networks/correlation_package
python3 setup.py install

cd ../resample2d_package
python3 setup.py install

cd ../channelnorm_package
python3 setup.py install

cd ..
