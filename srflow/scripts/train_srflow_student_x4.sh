DATASET_PATH="../data"
CONFIG_PATH="../confs/SRFlow_DF2K_4X.yml"
EXPERIMENT_PATH="../logdir/SRFlow_Student_X4"

CUDA_VISIBLE_DEVICES=0 python ../distill_srflow.py \
    --config $CONFIG_PATH \
    --logdir $EXPERIMENT_PATH \
    --dataset $DATASET_PATH \
    --num-epochs 100 \
    --eval-interval 1000 \
    --ckpt-interval 20 \
    --train-info-interval 500 \
    --patch-size 128 \
    --alpha 10.0 \
    --K 6 \
    --learning-rate 2e-4 \
    --batch-size 80 \
    --use-pretrained-lr-encoder