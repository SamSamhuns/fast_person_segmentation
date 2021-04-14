#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate server_c # this should be replaced wit the right conda env
python inference_smoothed_with_slider.py    media/img/beach.jpg &
python inference_nosmoothed_with_slider.py  media/img/beach.jpg &
python inference_tensorflow_pb.py           media/img/beach.jpg &
