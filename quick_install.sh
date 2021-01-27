#!/bin/bash

function install_package(){
    for dn in "build" "dist" "*.egg-info";do
        if [ -d $dn ];then
            rm -r $dn
        fi
    done || exit 1;

    #python3 setup.py install
    pip3 install -U pip
    pip3 install wheel setuptools
    pip3 install numpy==1.16
    python3 setup.py sdist bdist_wheel && cd dist && pip3 install *.whl || exit 1;

    cd ..    
    # rm -r build dist *.egg-info
}

echo y | pip3 uninstall exkaldi;
python3 -c "import kenlm" 2>/dev/null || {
    cd src && cd kenlm || exit 1;
    install_package
    cd ../..
}

install_package
