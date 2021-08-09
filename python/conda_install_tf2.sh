#!/bin/bash
# create a conda env `server_c` required for train, test, and inference

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "darwin system conda install starting for GPU mode"
  yes | conda create -n fastseg python=3.7
  eval "$(conda shell.bash hook)"
  conda activate fastseg
  yes | pip install tensorflow-gpu==2.5.0
  yes | pip install openvino==2021.3.0
  yes | pip install opencv-python==4.5.3.56
  yes | pip install scipy==1.7.1
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "darwin system conda install starting (Only avai in CPU)"
  yes | conda create -n fastseg python=3.7
  eval "$(conda shell.bash hook)"
  conda activate fastseg
  yes | pip install tensorflow==2.5.0
  yes | pip install openvino==2021.3.0
  yes | pip install opencv-python==4.5.3.56
  yes | pip install scipy==1.7.1
elif [[ "$OSTYPE" == "cygwin" ]]; then
  echo "cygwin system conda install not implemented"
        # POSIX compatibility layer and Linux environment emulation for Windows
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
