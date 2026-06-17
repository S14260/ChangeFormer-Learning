"""
用 LEVIR 预训练权重微调 tianjin_wayback 数据集（增强版）

改进点：
1. 数据增强：随机旋转 + 更强的颜色抖动 + 更大的裁剪范围
2. 加权 Loss：变化类权重 3.0，解决正负样本不均衡
3. Label Smoothing：0.1，缓解过拟合
4. Cosine LR：训练更稳定

用法:
  python finetune_LEVIR.py
"""

import subprocess
import sys
import os

def main():
    config = {
        # 数据
        "data_name": "tianjin_wayback",
        "img_size": 256,
        "batch_size": 4,
        "split": "train",
        "split_val": "val",

        # 模型
        "net_G": "ChangeFormerV6",
        "embed_dim": 256,
        "n_class": 2,

        # 微调参数
        "lr": 0.00006,
        "max_epochs": 50,
        "optimizer": "adamw",
        "loss": "ce",
        "lr_policy": "cosine",         # 改为 cosine 调度

        # 预训练权重
        "pretrain": "./checkpoints/ChangeFormer_LEVIR/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt",

        # 增强参数
        "multi_scale_train": True,
        "multi_scale_infer": False,
        "shuffle_AB": False,
        "gpu_ids": "0",

        # 新增：加权 Loss [背景权重, 变化类权重]
        "loss_weight": [1.0, 3.0],

        # 新增：Label Smoothing
        "label_smoothing": 0.1,

        # 输出目录
        "checkpoint_root": "./checkpoints",
        "vis_root": "./vis",
        "project_name": "finetune_tianjin_wayback_from_LEVIR_enhanced_e50_b4",
    }

    cmd = [
        sys.executable, "main_cd.py",
        f"--data_name", config["data_name"],
        f"--img_size", str(config["img_size"]),
        f"--batch_size", str(config["batch_size"]),
        f"--split", config["split"],
        f"--split_val", config["split_val"],
        f"--net_G", config["net_G"],
        f"--embed_dim", str(config["embed_dim"]),
        f"--lr", str(config["lr"]),
        f"--max_epochs", str(config["max_epochs"]),
        f"--optimizer", config["optimizer"],
        f"--loss", config["loss"],
        f"--lr_policy", config["lr_policy"],
        f"--pretrain", config["pretrain"],
        f"--gpu_ids", config["gpu_ids"],
        f"--checkpoint_root", config["checkpoint_root"],
        f"--vis_root", config["vis_root"],
        f"--project_name", config["project_name"],
    ]

    # 布尔参数
    if config["multi_scale_train"]:
        cmd.extend(["--multi_scale_train", "True"])
    if config["multi_scale_infer"]:
        cmd.extend(["--multi_scale_infer", "True"])
    if config["shuffle_AB"]:
        cmd.extend(["--shuffle_AB", "True"])

    # 加权 Loss
    if config["loss_weight"]:
        cmd.extend(["--loss_weight"] + [str(w) for w in config["loss_weight"]])

    # Label Smoothing
    if config["label_smoothing"]:
        cmd.extend(["--label_smoothing", str(config["label_smoothing"])])

    print("=" * 60)
    print("LEVIR→tianjin_wayback 增强微调配置")
    print("=" * 60)
    print(f"预训练权重: {config['pretrain']}")
    print(f"学习率: {config['lr']}")
    print(f"LR策略: {config['lr_policy']}")
    print(f"Epoch: {config['max_epochs']}")
    print(f"Loss权重: {config['loss_weight']}")
    print(f"Label Smoothing: {config['label_smoothing']}")
    print(f"数据增强: 旋转+颜色抖动+随机裁剪+翻转+模糊")
    print(f"输出目录: checkpoints/{config['project_name']}")
    print("=" * 60)
    print(f"\n执行命令:\n{' '.join(cmd)}\n")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
