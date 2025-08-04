#! /bin/bash

if [ ! -d "build" ]; then
    mkdir build
fi

cd build
cmake ../
make
cd ..

python3 test_hlo.py
