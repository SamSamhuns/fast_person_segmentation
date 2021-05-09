#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate server_c      # replace with the right conda env

export PYTHONPATH=${PYTHONPATH}:./
export PYTHONPATH=${PYTHONPATH}:./utils/

INFERENCE_PROG=$1            # inference.py program
shift                        # del 1st arg
python3 $INFERENCE_PROG "$@" # python3 inference_minimal.py -b media/img/beach.jpg
