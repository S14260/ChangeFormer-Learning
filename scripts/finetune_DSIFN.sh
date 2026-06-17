#!/usr/bin/env bash

# 用 LEVIR 预训练权重微调 DSIFN 数据集
# 用法: bash scripts/finetune_DSIFN.sh

#GPUs
gpus=0

#Set paths
checkpoint_root=./checkpoints
vis_root=./vis
data_name=DSIFN

# 微调参数（关键：小学习率，少 epoch）
img_size=256
batch_size=16
lr=0.00006          # 微调用小学习率（从头训练用 0.0001）
max_epochs=50       # 微调收敛快，50 epoch 够了
embed_dim=256

net_G=ChangeFormerV6

lr_policy=linear
optimizer=adamw
loss=ce
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

# 用 LEVIR 预训练权重初始化
pretrain=./checkpoints/ChangeFormer_LEVIR/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt

#Train and Validation splits
split=train
split_val=val
project_name=finetune_DSIFN_from_LEVIR_lr${lr}_e${max_epochs}

python main_cd.py \
    --img_size ${img_size} \
    --loss ${loss} \
    --checkpoint_root ${checkpoint_root} \
    --vis_root ${vis_root} \
    --lr_policy ${lr_policy} \
    --optimizer ${optimizer} \
    --pretrain ${pretrain} \
    --split ${split} \
    --split_val ${split_val} \
    --net_G ${net_G} \
    --multi_scale_train ${multi_scale_train} \
    --multi_scale_infer ${multi_scale_infer} \
    --gpu_ids ${gpus} \
    --max_epochs ${max_epochs} \
    --project_name ${project_name} \
    --batch_size ${batch_size} \
    --shuffle_AB ${shuffle_AB} \
    --data_name ${data_name} \
    --lr ${lr} \
    --embed_dim ${embed_dim}
