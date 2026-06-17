"""

用 DSIFN 预训练权重微调 tianjin_wayback 数据集

用法:
  python finetune_DSIFN.py
"""

import subprocess
import sys
import os

def main():
    # 微调参数（针对 1080Ti 11GB 优化）
    config = {
        # 数据
        "data_name": "tianjin_wayback",
        "img_size": 256,
        "batch_size": 4,         # 11GB 显存，可以从 8 提升到 16
        "split": "train",
        "split_val": "val",

        # 模型
        "net_G": "ChangeFormerV6",
        "embed_dim": 256,
        "n_class": 2,

        # 微调关键参数
        "lr": 0.00006,           # 小学习率
        "max_epochs": 50,        # 少 epoch
        "optimizer": "adamw",
        "loss": "ce",
        "lr_policy": "linear",

        # 预训练权重
        "pretrain": "./checkpoints/ChangeFormer_DSIFN/CD_ChangeFormerV6_DSIFN_b16_lr0.00006_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt",

        # 其他（1080Ti 11GB 可以开启多尺度训练）
        "multi_scale_train": True,   # 显存足够，开启多尺度训练提升效果
        "multi_scale_infer": False,
        "shuffle_AB": False,
        "gpu_ids": "0",

        # 输出目录
        "checkpoint_root": "./checkpoints",
        "vis_root": "./vis",
        "project_name": "finetune_tianjin_wayback_from_DSIFN_lr0.00006_e50_b16",
    }

    # 构建命令
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

    # 布尔参数处理
    if config["multi_scale_train"]:
        cmd.extend(["--multi_scale_train", "True"])
    if config["multi_scale_infer"]:
        cmd.extend(["--multi_scale_infer", "True"])
    if config["shuffle_AB"]:
        cmd.extend(["--shuffle_AB", "True"])

    # 打印配置
    print("=" * 60)
    print("DSIFN→tianjin_wayback 微调配置")
    print("=" * 60)
    print(f"预训练权重: {config['pretrain']}")
    print(f"学习率: {config['lr']}")
    print(f"Epoch: {config['max_epochs']}")
    print(f"输出目录: checkpoints/{config['project_name']}")
    print("=" * 60)
    print(f"\n执行命令:\n{' '.join(cmd)}\n")

    # 执行训练
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
