#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
# Your tcmalloc .so path
export LD_PRELOAD="/usr/lib/libtcmalloc.so"
train_dir="./group_2-2-2"
# Our train/val split dataset
# Run 'python download_cifar100.py' before training
data_dir="./cifar100/train_val_split"

python train.py --train_dir $train_dir \
    --data_dir $data_dir \
    --batch_size 90 \
    --test_interval 500 \
    --test_iter 50 \
    --num_residual_units 2 \
    --k 8 \
    --ngroups1 2 \
    --ngroups2 2 \
    --ngroups3 2 \
    --l2_weight 0.0001 \
    --gamma1 1.0 \
    --gamma2 1.0 \
    --gamma3 10.0 \
    --dropout_keep_prob 0.5 \
    --initial_lr 0.1 \
    --lr_step_epoch 240.0,300.0 \
    --lr_decay 0.1 \
    --bn_no_scale True \
    --weighted_group_loss True \
    --max_steps 200000 \
    --checkpoint_interval 5000 \
    --group_summary_interval 5000 \
    --gpu_fraction 0.96 \
    --display 100 \
    #--checkpoint "./group_2-2-2/model.ckpt-149999" \
    #--finetune True \
    #--basemodel "./baseline/model.ckpt-449999" \
    #--load_last_layers True \


# Deep split(2-2-2)
# Dropout with prob 0.5
# Softmax reparametrization & Training from scratch
