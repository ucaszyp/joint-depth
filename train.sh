CUDA_VISIBLE_DEVICES=0,1,2,3
python3 train.py \
--config rnw_ns \
--gpus 4 \
--work_dir debug_check
