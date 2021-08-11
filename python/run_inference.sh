#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate fastseg       # replace with the right conda env

export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
export PYTHONPATH=${PYTHONPATH}:./
export PYTHONPATH=${PYTHONPATH}:./utils/

INFERENCE_PROG=$1            # inference.py program
shift                        # del 1st arg
python3 $INFERENCE_PROG "$@" # python3 inference_minimal.py -b media/img/beach.jpg
