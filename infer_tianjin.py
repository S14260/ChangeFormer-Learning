"""
ChangeFormer 通用推理脚本
支持任意预训练模型 + 任意数据目录

用法:
  # 用 LEVIR 预训练模型推理 tianjin_cd 数据集
  python infer_tianjin.py --model levir

  # 用 DSIFN 预训练模型推理 tianjin_cd 数据集
  python infer_tianjin.py --model dsifn

  # 用 LEVIR 模型推理自定义数据目录
  python infer_tianjin.py --model levir --data_dir ./your_data

  # 用 DSIFN 模型推理 wayback 数据集
  python infer_tianjin.py --model dsifn --data_name tianjin_wayback

  # 指定输出目录
  python infer_tianjin.py --model levir --output my_output
"""
from argparse import ArgumentParser
import utils
import torch
from torch.utils.data import DataLoader
from datasets.CD_dataset import CDDataset
from models.basic_model import CDEvaluator
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 预训练模型配置
MODEL_CONFIGS = {
    'levir': {
        'project_name': 'CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256',
        'checkpoint_root': r'E:\changeformer\ChangeFormer\checkpoints\ChangeFormer_LEVIR',
    },
    'dsifn': {
        'project_name': 'CD_ChangeFormerV6_DSIFN_b16_lr0.00006_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256',
        'checkpoint_root': r'E:\changeformer\ChangeFormer\checkpoints\ChangeFormer_DSIFN',
    },
    'finetune_LEVIR': {
        'project_name': 'finetune_tianjin_wayback_from_LEVIR_lr0.00006_e50_b4',
        'checkpoint_root': r'E:\changeformer\ChangeFormer\checkpoints',
    },
    'finetune_DSIFN': {
        'project_name': 'finetune_tianjin_wayback_from_DSIFN_lr0.00006_e50_b4',
        'checkpoint_root': r'E:\changeformer\ChangeFormer\checkpoints',
    },
    'finetune_LEVIR_enhanced': {
        'project_name': 'finetune_tianjin_wayback_from_LEVIR_enhanced_e50_b4',
        'checkpoint_root': r'E:\changeformer\ChangeFormer\checkpoints',
    },
}


def get_args():
    parser = ArgumentParser(description='ChangeFormer 通用推理脚本')
    parser.add_argument('--model', type=str, default='levir', choices=['levir', 'dsifn', 'finetune_LEVIR', 'finetune_DSIFN','finetune_LEVIR_enhanced'],
                        help='预训练模型: levir, dsifn 或 finetune_LEVIR 或 finetune_DSIFN 或 finetune_LEVIR_enhanced')
    parser.add_argument('--data_name', type=str, default='tianjin_wayback',
                        help='数据集名称 (通过 data_config.py 查找，如 tianjin_cd, tianjin_wayback)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='自定义数据目录 (优先于 data_name，需包含 A/, B/, list/demo.txt)')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='输出目录 (默认: output_tianjin_{model})')
    parser.add_argument('--split', type=str, default='train', help='数据集 split (train/val/demo)')
    parser.add_argument('--img_size', type=int, default=256, help='输入图片大小')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID, -1 为 CPU')
    parser.add_argument('--checkpoint_name', type=str, default='best_ckpt.pt')
    args = parser.parse_args()

    # 根据 model 选择配置
    cfg = MODEL_CONFIGS[args.model]
    args.project_name = cfg['project_name']
    args.checkpoint_root = cfg['checkpoint_root']
    args.n_class = 2
    args.embed_dim = 256
    args.net_G = 'ChangeFormerV6'
    args.num_workers = 0

    # 默认输出目录
    if args.output_folder is None:
        args.output_folder = f'output_tianjin_{args.model}'

    return args


if __name__ == '__main__':
    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                          else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"模型: {args.model.upper()}")
    print(f"设备: {device}")
    print(f"检查点: {args.checkpoint_dir}")
    print(f"输出: {os.path.abspath(args.output_folder)}")

    # 加载数据
    if args.data_dir:
        # 直接指定目录
        dataset = CDDataset(root_dir=args.data_dir, split=args.split,
                            img_size=args.img_size, is_train=False)
        data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
        print(f"数据目录: {args.data_dir}")
    else:
        # 通过 data_config.py 查找
        data_loader = utils.get_loader(
            args.data_name, img_size=args.img_size,
            batch_size=args.batch_size,
            split=args.split, is_train=False
        )
        print(f"数据集: {args.data_name}")

    # 加载模型
    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    print(f"开始推理, 共 {len(data_loader)} 对影像...")
    for i, batch in enumerate(data_loader):
        name = batch['name']
        print(f'[{i+1}/{len(data_loader)}] {name}')
        model._forward_pass(batch)
        model._save_predictions()

    print(f"\n推理完成! 结果保存在: {os.path.abspath(args.output_folder)}")
