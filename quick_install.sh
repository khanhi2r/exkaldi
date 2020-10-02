#!/bin/bash

function install_package(){

    for dn in "build" "dist" "*.egg-info";do
        if [ -d $dn ];then
            rm -r $dn
        fi
    done || exit 1;

    python3 setup.py sdist bdist_wheel && cd dist && pip install * || exit 1;

    cd ..

    rm -r build dist *.egg-info

}

echo y | pip uninstall exkaldi;

echo n | pip uninstall kenlm || {

    cd src && cd kenlm || exit 1;

    install_package

    cd ../..

}

install_package
