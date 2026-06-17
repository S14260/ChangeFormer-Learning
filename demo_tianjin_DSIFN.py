"""
使用 LEVIR 预训练的 ChangeFormerV6 对天津自然保护地影像进行推理
输出: 变化检测二值图 (0=无变化, 255=有变化)
"""
from argparse import ArgumentParser
import utils
import torch
from models.basic_model import CDEvaluator
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--project_name',
                        default='CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256',
                        type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: use -1 for CPU')
    parser.add_argument('--checkpoint_root',
                        default=r'E:\changeformer\ChangeFormer\checkpoints\ChangeFormer_LEVIR',
                        type=str)
    parser.add_argument('--output_folder', default='output_tianjin', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='tianjin_cd', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='ChangeFormerV6', type=str)
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                          else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    print(f"使用设备: {device}")
    print(f"检查点: {args.checkpoint_dir}")
    print(f"输出目录: {args.output_folder}")

    data_loader = utils.get_loader(
        args.data_name, img_size=args.img_size,
        batch_size=args.batch_size,
        split=args.split, is_train=False
    )

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    print(f"开始推理, 共 {len(data_loader)} 对影像...")
    for i, batch in enumerate(data_loader):
        name = batch['name']
        print(f'[{i+1}/{len(data_loader)}] {name}')
        score_map = model._forward_pass(batch)
        model._save_predictions()

    print(f"\n推理完成! 结果保存在: {os.path.abspath(args.output_folder)}")
