WORKDIR=h_mask_aug_7
mkdir $WORKDIR
cp -r configs/rnw_ns.yaml  $WORKDIR
cp -r models/rnw.py $WORKDIR
cp -r train.sh $WORKDIR
cp -r train.py $WORKDIR
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8 \
python3 train.py \
--config rnw_ns \
--gpus 7 \
--work_dir $WORKDIR