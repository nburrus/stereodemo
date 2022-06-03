#!/bin/bash

rm -f stereodemo/datasets

cd stereodemo
ln -sf ../datasets .
cd ..

rm -f dist/*
python3 -m build

rm -f stereodemo/datasets