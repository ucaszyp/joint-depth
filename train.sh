WORKDIR=mask-illu-aug-1
mkdir $WORKDIR
cp -r configs/rnw_ns.yaml  $WORKDIR
cp -r models/rnw.py $WORKDIR
cp -r train.sh $WORKDIR
cp -r train.py $WORKDIR
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train.py \
--config rnw_ns \
--gpus 4 \
--work_dir $WORKDIR

