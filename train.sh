CUDA_VISIBLE_DEVICES=4,5,6,7 \
python3 train.py \
--config rnw_ns \
--gpus 4 \
--work_dir joint-bs16-s-5
