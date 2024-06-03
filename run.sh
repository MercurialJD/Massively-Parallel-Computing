#!/bin/bash

clean() {
    echo "Cleaning up..."
    ./cleanfile.sh
    rm -rf ./m1 ./m2 ./m3 ./final *.out *.err outfile
}

build() {
    echo "Building the project..."
    ./cleanfile.sh
    cmake -DCMAKE_CXX_FLAGS=-pg ./project/ && make -j8      # For scratch build
    # cmake ./project/ && make -j8                            # For profile build
    ./cleanfile.sh
}


case "$1" in
    clean) clean ;;
    build) build ;;
    *) echo "Usage: $0 {clean|build}" ;;
esac
