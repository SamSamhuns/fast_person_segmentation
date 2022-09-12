#!/bin/bash
# create a conda env `server_c` required for train, test, and inference

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "darwin system conda install starting for GPU mode"
  yes | conda create -n fastseg python=3.7
  eval "$(conda shell.bash hook)"
  conda activate fastseg
  yes | pip install -r requirements.txt
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "darwin system conda install starting (Only avai in CPU)"
  yes | conda create -n fastseg python=3.7
  eval "$(conda shell.bash hook)"
  conda activate fastseg
  yes | pip install -r requirements.txt
elif [[ "$OSTYPE" == "cygwin" ]]; then
  yes | conda create -n fastseg python=3.7
  eval "$(conda shell.bash hook)"
  conda activate fastseg
  yes | pip install -r requirements.txt
elif [[ "$OSTYPE" == "msys" ]]; then
  echo "msys system conda install not implemented"
        # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
elif [[ "$OSTYPE" == "win32" ]]; then
  echo "win32 system conda install not implemented"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
  echo "freebsd system conda install not implemented"
else
  echo "system unknown"
fi
