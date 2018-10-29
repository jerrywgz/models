#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
cudaid=${object_detection_cudaid:=0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=$cudaid
python train.py --max_iter=3000 --enable_ce=True | python _ce.py

