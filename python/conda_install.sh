#!/bin/bash


if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "darwin system conda install starting"
  yes | conda create -n server_c python=3.7
  eval "$(conda shell.bash hook)"
  conda activate server_c
  yes | conda install -c conda-forge tensorflow=1.14.0;
  yes | conda install -c conda-forge keras=2.2.4;
  yes | conda install -c conda-forge opencv;
  yes | conda install -c anaconda h5py=2.10.0;
  yes | conda install -c anaconda tensorflow-gpu=1.14.0;
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "darwin system conda install starting"
  yes | conda create -n server_c python=3.7
  eval "$(conda shell.bash hook)"
  conda activate server_c
  yes | conda install -c conda-forge opencv;
  yes | conda install -c conda-forge tensorflow=1.14.0;
  yes | conda install -c conda-forge keras=2.2.4;
  yes | conda install -c anaconda h5py=2.10.0;
  yes | conda install -c anaconda tensorflow-gpu=1.14.0;
elif [[ "$OSTYPE" == "cygwin" ]]; then
  echo "cygwin system conda install not implmented"
        # POSIX compatibility layer and Linux environment emulation for Windows
elif [[ "$OSTYPE" == "msys" ]]; then
  echo "msys system conda install not implmented"
        # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
elif [[ "$OSTYPE" == "win32" ]]; then
  echo "win32 system conda install not implmented"
elif [[ "$OSTYPE" == "freebsd"* ]]; then
  echo "freebsd system conda install not implmented"
else
  echo "system unknown"
fi
