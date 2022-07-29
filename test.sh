#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python test_nuscenes_disp.py night rnw_ns joint-illu-13/rnw_ns/checkpoint_epoch=17.ckpt
cd evaluation
python eval_nuscenes.py night