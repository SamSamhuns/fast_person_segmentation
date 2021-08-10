#!/bin/bash
# set flags for tensorflow training

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
# Default 0: show all logs. 1: filter INFO logs. 2: filter WARNING. 3: filter ERROR
export TF_CPP_MIN_LOG_LEVEL="1"
export OMP_NUM_THREADS="15"
export KMP_BLOCKTIME="0"
export KMP_SETTINGS="1"
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
# PCI_BUS_ID ensures cuda device order matches inside tf
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# specify which GPU(s) to be used
export CUDA_VISIBLE_DEVICES="1"

# run train script
python3 train_munet_mnetv2.py
