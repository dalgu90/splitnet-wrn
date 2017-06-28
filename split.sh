#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
train_dir="./split_2-2-2"
#data_dir="./cifar100/train_val_split"
data_dir="/data1/dalgu/cifar100/train_val_split"

python train_split.py --train_dir $train_dir \
    --data_dir $data_dir \
    --batch_size 90 \
    --test_interval 500 \
    --test_iter 50 \
    --num_residual_units 2 \
    --k 8 \
    --ngroups1 2 \
    --ngroups2 2 \
    --ngroups3 2 \
    --l2_weight 0.0005 \
    --initial_lr 0.1 \
    --lr_step_epoch 100.0,140.0 \
    --lr_decay 0.1 \
    --max_steps 120000 \
    --checkpoint_interval 5000 \
    --gpu_fraction 0.96 \
    --display 100 \
    --basemodel "./group_2-2-2/model.ckpt-199999" \
    #--checkpoint "./split_2-2-2/model.ckpt-40000" \
    #--finetune True \
    #--load_last_layers True \


# Finetune with Deep split
