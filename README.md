# ChangeFormer: 基于Transformer的遥感影像变化检测

> 原论文: [A Transformer-Based Siamese Network for Change Detection](https://arxiv.org/abs/2201.01293) (IGARSS 2022)

## 项目简介

基于 ChangeFormer 预训练模型，对**天津 Wayback 历史遥感影像**进行建筑变化检测的迁移学习与微调。

### 应用场景
- 城市建设监测
- 违建识别
- 农用地变化检测
- 灾害评估

---

## 改进内容（相对于原 ChangeFormer）

### 1. Bug 修复
- 修复数据增强裁剪时 label 填充值错误（255→0），避免裁剪区域被误标为"变化"

### 2. 数据增强
- 随机旋转（90°/180°/270°）
- 水平/垂直翻转
- 颜色抖动（亮度/对比度/饱和度 ±0.5）
- 高斯模糊
- 随机裁放（0.8x ~ 1.3x）

### 3. 加权 CrossEntropy Loss
- 变化类权重设为 3.0，解决正负样本不均衡（变化像素远少于背景）
- 通过 `--loss_weight 1.0 3.0` 参数控制

### 4. Label Smoothing
- 设置为 0.1，软化标签防止过拟合
- 通过 `--label_smoothing 0.1` 参数控制

### 5. Cosine Annealing LR
- 学习率从初始值平滑衰减到 1e-7，训练后期更稳定
- 通过 `--lr_policy cosine` 参数控制

---

## 实验结果

### 数据集
- **训练集**: 105 对影像（LabelMe 手动标注）
- **验证集**: 14 对影像
- **测试集**: 32 对影像

### 模型对比（测试集）

| 模型 | 预训练数据 | mF1 | 变化类 F1 | Precision | Recall |
|------|-----------|-----|----------|-----------|--------|
| LEVIR 微调（原始） | LEVIR-CD (637对) | 0.761 | 0.556 | 0.794 | 0.428 |
| DSIFN 微调 | DSIFN-CD (3940对) | 0.689 | 0.417 | 0.799 | 0.282 |
| **LEVIR 微调（增强版）** | LEVIR-CD (637对) | **0.782** | **0.601** | 0.650 | **0.560** |

> 增强版相比原始 LEVIR 微调：mF1 +2.1%，变化类 F1 +4.5%，Recall +31%（0.428→0.560）

### 训练过程对比（增强版）

| 指标 | Epoch 0（初始） | Best (Epoch 42) | 提升 |
|------|----------------|-----------------|------|
| mF1 | 0.528 | 0.679 | +29% |
| 变化类 F1 | 0.160 | 0.395 | +147% |
| 变化类 Precision | 0.120 | 0.435 | +263% |
| 变化类 Recall | 0.238 | 0.362 | +52% |
| 整体 Accuracy | 0.817 | 0.931 | +14% |

### 可视化结果

<!-- TODO: 替换为实际图片 -->
![LEVIR vs DSIFN 对比](./images/comparison_levir_vs_dsifn.png)

![训练曲线](./images/training_curve.png)

![预测结果示例](./images/prediction_examples.png)

---

## 快速开始

### 环境要求
```
Python 3.8+
PyTorch 1.10+
torchvision
einops
pillow
numpy
```

### 安装
```bash
conda create --name ChangeFormer python=3.8
conda activate ChangeFormer
pip install torch torchvision einops pillow numpy
```

---

## 使用方法

### 1. 微调训练

```bash
# LEVIR 预训练权重微调（推荐，增强版）
python finetune_LEVIR.py

# DSIFN 预训练权重微调
python finetune_DSIFN.py
```

### 2. 推理

```bash
# 用增强版 LEVIR 微调模型推理测试集
python infer_tianjin.py --model finetune_LEVIR --data_name tianjin_wayback --split test --output_folder output_finetune_LEVIR_test

# 用 DSIFN 微调模型推理测试集
python infer_tianjin.py --model finetune_DSIFN --data_name tianjin_wayback --split test --output_folder output_finetune_DSIFN_test

# 用原版 LEVIR 预训练模型推理
python infer_tianjin.py --model levir --data_name tianjin_wayback --split test
```

### 3. 结果对比

```bash
# 生成 LEVIR vs DSIFN 预测对比图
python compare_models.py
```

---

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_name` | tianjin_wayback | 数据集名称 |
| `--batch_size` | 4 | 批次大小 |
| `--lr` | 0.00006 | 学习率（微调用小学习率） |
| `--max_epochs` | 50 | 训练轮数 |
| `--embed_dim` | 256 | Transformer 嵌入维度 |
| `--net_G` | ChangeFormerV6 | 网络版本 |
| `--optimizer` | adamw | 优化器 |
| `--loss` | ce | 损失函数 |
| `--loss_weight` | 1.0 3.0 | 类别权重 [背景, 变化] |
| `--label_smoothing` | 0.1 | 标签平滑系数 |
| `--lr_policy` | cosine | 学习率策略 (linear/cosine/step) |

---

## 数据集结构

```
datasets/tianjin_wayback/
├── train/
│   ├── A/          # 前期影像
│   ├── B/          # 后期影像
│   ├── label/      # 二值标签（0=背景, 255=变化）
│   └── list/
│       └── train.txt
├── val/
│   ├── A/, B/, label/
│   └── list/val.txt
└── test/
    ├── A/, B/, label/
    └── list/test.txt
```

### 标注工具
使用 [LabelMe](https://github.com/labelmeai/labelme) 进行多边形标注，标注后用 `convert_labelme_to_mask.py` 转换为二值 mask。

---

## 项目结构

```
ChangeFormer/
├── models/
│   ├── ChangeFormer.py      # ChangeFormer 网络定义
│   ├── trainer.py           # 训练器（支持加权Loss、Label Smoothing）
│   ├── losses.py            # 损失函数（支持 label_smoothing）
│   └── networks.py          # 网络构建（支持 cosine LR）
├── datasets/
│   ├── CD_dataset.py        # 数据集类（支持旋转增强）
│   ├── data_utils.py        # 数据增强工具（已修复裁剪bug）
│   └── tianjin_wayback/     # 天津 Wayback 数据集
├── main_cd.py               # 训练入口
├── finetune_LEVIR.py        # LEVIR 增强微调脚本（推荐）
├── finetune_DSIFN.py        # DSIFN 微调脚本
├── infer_tianjin.py         # 通用推理脚本
├── compare_models.py        # 模型对比可视化
├── data_config.py           # 数据集配置
└── checkpoints/             # 模型权重（已 gitignore）
```

---

## 引用

如果本项目对您有帮助，请引用原论文：

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

## 许可证

本代码仅用于非商业和研究目的。商业用途请联系原作者。

## 致谢

- [ChangeFormer](https://github.com/wgcban/ChangeFormer) - 原始实现
- [BIT_CD](https://github.com/justchenhao/BIT_CD) - ChangeFormer 基于此代码库实现
