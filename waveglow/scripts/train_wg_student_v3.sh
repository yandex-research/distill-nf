DATASET_PATH="./data/wavs"
TEACHER_PATH="./pretrained_models/wg_teacher_ch256_wn12.pth"
EXPERIMENT_PATH="./logdir/wg_student_v3"

python -m torch.distributed.launch --nproc_per_node=8 distill_waveglow.py \
    --logdir $EXPERIMENT_PATH \
    --dataset $DATASET_PATH \
    --teacher-path $TEACHER_PATH \
    --num-steps 200000 \
    --eval-interval 2000 --train-info-interval 500 \
    --segment-length 16000 \
    --batch-size 32 \
    --stft-loss-coeff 0.02 \
    --student_n_wavenets 2 \
    --student_hid_channels 96 \
    --onecycle \
    --onecycle-lr-base 1e-3 \
    --onecycle-warmup-steps 5000 \
    --onecycle-decay-rate 0.2 \
    --onecycle-lr-min 1e-5
