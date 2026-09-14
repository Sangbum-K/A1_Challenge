#!/bin/sh
## Copyright 2019 Alexander Liniger

## Install dependencies
set -e

cd ../../../lib/hpipm

if [ ! -d "blasfeo/build" ] || [ ! -d "blasfeo/lib" ]; then
    rm -rf blasfeo/build
    rm -rf blasfeo/lib
    cd blasfeo
    mkdir -p build
    mkdir -p lib
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(realpath ../lib) -DBLASFEO_EXAMPLES=OFF 
    make
    make install
    cd ../..
fi

echo "install blasfeo"

if [ ! -d "hpipm/build" ] || [ ! -d "hpipm/lib" ]; then
    rm -rf hpipm/build
    rm -rf hpipm/lib
    cd hpipm
    mkdir -p build
    mkdir -p lib
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(realpath ../lib) -DBLASFEO_PATH=$(realpath ../../blasfeo/lib) -DHPIPM_TESTING=OFF 
    make
    make install
    cd ../../../
fi
echo "install hpipm"
