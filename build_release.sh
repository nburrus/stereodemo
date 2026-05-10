#!/bin/zsh

set -e

rm -rf stereodemo/datasets
mkdir -p stereodemo/datasets
trap 'rm -rf stereodemo/datasets' EXIT

ln -sf ../../datasets/oak-d stereodemo/datasets/oak-d

rm -f dist/*
uv build
