#!/bin/bash

dir=$(dirname "$0")
if [ "$1" = "shell" ]; then
    PYTHONPATH="$dir/lib/" ipython
else
    PYTHONPATH="$dir/lib/" python3.6 "$@"
fi
