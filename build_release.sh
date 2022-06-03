#!/bin/zsh

rm -rf stereodemo/datasets
mkdir stereodemo/datasets

pushd stereodemo/datasets
ln -sf ../../datasets/oak-d .
popd

rm -f dist/*
python3 -m build

rm -rf stereodemo/datasets
