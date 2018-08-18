#!/bin/bash

dir=$(dirname "$0")
PYTHONPATH="$dir/lib/" python3.6 "$@"
