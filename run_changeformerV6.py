from argparse import ArgumentParser

import utils
import torch
from torch.utils.data import DataLoader
from datasets.CD_dataset import CDDataset
from models.basic_model import CDEvaluator

import os

"""
用自己训练的 ChangeFormerV6 模型做推理

用法:
  python run_changeformerV6.py --data_dir ./samples_LEVIR
  python run_changeformerV6.py --data_dir ./your_real_data

数据目录结构:
  data_dir/
  ├── A/              # 时相1图片
  ├── B/              # 时相2图片（与A同名）
  └── list/
      └── demo.txt    # 每行一个文件名
"""


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--project_name', default='LEVIR_ChangeFormerV6', type=str)
    #parser.add_argument('--project_name', default='ChangeFormer_LEVIR/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256', type=str)
    parser.add_argument('--output_folder', default='vis/LEVIR_ChangeFormerV6/predict', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--data_dir', default='./samples_LEVIR', type=str,
                        help='数据目录，需包含 A/, B/, list/demo.txt')

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=128, type=int)
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

    # 直接用 data_dir 构建数据集，不经过 data_config.py
    dataset = CDDataset(root_dir=args.data_dir, split=args.split,
                        img_size=args.img_size, is_train=False)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        score_map = model._forward_pass(batch)
        model._save_predictions()

    print('Done! Predictions saved to: %s' % args.output_folder)
