#!/bin/bash

set -e

rm -rf stereodemo/datasets
mkdir -p stereodemo/datasets
trap 'rm -rf stereodemo/datasets' EXIT

ln -sf ../../datasets/oak-d stereodemo/datasets/oak-d

mkdir -p dist
rm -f dist/*
uv build
