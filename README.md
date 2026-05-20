# ChangeFormer: 基于Transformer的变化检测网络

> 论文: [A Transformer-Based Siamese Network for Change Detection](https://arxiv.org/abs/2201.01293)

> 作者: [Wele Gedara Chaminda Bandara](https://www.wgcban.com/) 和 [Vishal M. Patel](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)

> 发表于 [IGARSS-22](https://www.igarss2022.org/default.php)，马来西亚吉隆坡

## 📖 项目简介

这是一个基于 **Transformer** 的遥感影像变化检测项目，能够识别两期卫星影像中的地物变化（如建筑物新增/拆除）。

### 适用场景
- 🏗️ 城市建设监测
- 🌾 农用地变化检测
- 🏘️ 违建识别
- 🌍 灾害评估

## 🔗 相关链接

- 📄 论文 (IEEE): https://ieeexplore.ieee.org/document/9883686
- 📄 论文 (ArXiv): https://arxiv.org/abs/2201.01293
- 🎬 演示视频: https://www.youtube.com/watch?v=SkiNoTrSmQM

## 🏗️ 网络架构

![ChangeFormer架构](./images/IGARS_ChangeFormer.jpeg)

## 📊 实验结果

![LEVIR-CD和DSIFN-CD结果](./images/IGARS_ChangeFormer-LEVIR_DSFIN_both.png)

---

## 🚀 快速开始

### 环境要求

```
Python 3.8.0
PyTorch 1.10.1
torchvision 0.11.2
einops 0.3.2
```

### 安装环境

```bash
# 创建conda环境
conda create --name ChangeFormerV5 --file requirements.txt
conda activate ChangeFormerV5
```

### 克隆项目

```bash
git clone https://github.com/S14260/ChangeFormer-Learning.git
cd ChangeFormer-Learning
```

---

## 📁 数据集准备

### 数据目录结构

```
数据集文件夹/
├─ A/          # t1时期影像
├─ B/          # t2时期影像
├─ label/      # 变化标签图（黑白）
└─ list/       # 文件列表
    ├─ train.txt   # 训练集文件名列表
    ├─ val.txt     # 验证集文件名列表
    └─ test.txt    # 测试集文件名列表
```

### 下载数据集

**LEVIR-CD-256 数据集**: [点击下载](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip)

**DSIFN-CD-256 数据集**: [点击下载](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip)

---

## 🎯 快速演示

### 在 LEVIR 数据集上测试

1. 下载预训练模型: [`Github-LEVIR-Pretrained`](https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip)

2. 将模型放置到 `checkpoints/ChangeFormer_LEVIR/` 目录

3. 运行演示:
```bash
python demo_LEVIR.py
```

4. 结果保存在 `samples_LEVIR/predict_CD_ChangeFormerV6/`

### 在 DSIFN 数据集上测试

1. 下载预训练模型: [`Github-DSIFN-Pretrained`](https://github.com/wgcban/ChangeFormer/releases/download/v0.1.0/CD_ChangeFormerV6_DSIFN_b16_lr0.00006_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256.zip)

2. 将模型放置到 `checkpoints/ChangeFormer_DSIFN/` 目录

3. 运行演示:
```bash
python demo_DSIFN.py
```

---

## 🏋️ 模型训练

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 16 | 批次大小 |
| `lr` | 0.0001 | 学习率 |
| `max_epochs` | 200 | 训练轮数 |
| `img_size` | 256 | 输入图像尺寸 |
| `embed_dim` | 256 | Transformer嵌入维度 |
| `net_G` | ChangeFormerV6 | 网络版本 |
| `optimizer` | adamw | 优化器 (sgd/adam/adamw) |
| `loss` | ce | 损失函数 (ce/fl/miou) |

### 训练命令

```bash
python main_cd.py \
    --data_name LEVIR \
    --batch_size 16 \
    --lr 0.0001 \
    --max_epochs 200 \
    --net_G ChangeFormerV6 \
    --optimizer adamw \
    --loss ce \
    --embed_dim 256
```

### 使用预训练权重加速收敛

```bash
# 下载预训练的SegFormer权重
wget https://www.dropbox.com/s/undtrlxiz7bkag5/pretrained_changeformer.pt

# 训练时指定预训练路径
python main_cd.py --pretrain path/to/pretrained_changeformer.pt ...
```

---

## 📈 模型评估

```bash
python eval_cd.py \
    --data_name LEVIR \
    --net_G ChangeFormerV6 \
    --split test \
    --checkpoint_name best_ckpt.pt
```

---

## 📂 项目结构

```
ChangeFormer/
├── models/
│   ├── ChangeFormer.py      # ChangeFormer网络定义
│   ├── trainer.py           # 训练器
│   ├── evaluator.py         # 评估器
│   └── networks.py          # 网络构建
├── main_cd.py               # 训练入口
├── eval_cd.py               # 评估入口
├── demo_LEVIR.py            # LEVIR演示脚本
├── demo_DSIFN.py            # DSIFN演示脚本
├── data_config.py           # 数据集配置
├── utils.py                 # 工具函数
├── checkpoints/             # 模型权重（不上传）
├── vis/                     # 可视化结果（不上传）
└── scripts/                 # 训练/评估脚本
```

---

## 🔧 本项目的改进

在原项目基础上，本项目做了以下改进：

### 1. 兼容性修复
- 修复 `np.str` 弃用问题（改为 `str`）
- 修复 PyTorch `weights_only` 参数警告
- 修复 `F.interpolate` 使用 `size` 替代 `scale_factor` 的兼容性问题

### 2. Windows 环境适配
- 修改文件路径为 Windows 格式
- 适配本地 checkpoint 存储路径

### 3. 添加中文注释
- 在训练流程中添加详细中文注释
- 帮助理解训练、验证、保存三个阶段

---

## 📝 引用

如果本项目对您的研究有帮助，请引用原论文：

```bibtex
@INPROCEEDINGS{9883686,
  author={Bandara, Wele Gedara Chaminda and Patel, Vishal M.},
  booktitle={IGARSS 2022 - 2022 IEEE International Geoscience and Remote Sensing Symposium},
  title={A Transformer-Based Siamese Network for Change Detection},
  year={2022},
  pages={207-210},
  doi={10.1109/IGARSS46834.2022.9883686}
}
```

---

## 📄 许可证

本代码仅用于非商业和研究目的。商业用途请联系原作者。

## 🙏 致谢

感谢以下项目的支持：
- [BIT_CD](https://github.com/justchenhao/BIT_CD) - ChangeFormer 基于此代码库实现
