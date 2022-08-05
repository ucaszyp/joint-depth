#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python test_nuscenes_disp.py night rnw_ns joint-bs16-s-0.5-norm1/rnw_ns/checkpoint_epoch=10.ckpt --test 1
cd evaluation
python eval_nuscenes.py night