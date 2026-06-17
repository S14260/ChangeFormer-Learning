import os
import random
import shutil
import numpy as np
from PIL import Image

random.seed(42)

# 路径
base = os.path.dirname(os.path.abspath(__file__))
src_A = os.path.join(base, "train", "A")
src_B = os.path.join(base, "train", "B")
src_label = os.path.join(base, "train", "label")

# 读取所有样本
all_files = sorted([f for f in os.listdir(src_label) if f.endswith(".png")])

# 按有无变化分层
has_change = []
no_change = []
for f in all_files:
    arr = np.array(Image.open(os.path.join(src_label, f)))
    if 255 in arr:
        has_change.append(f)
    else:
        no_change.append(f)

# 分层划分：70% train, 10% val, 20% test
def split(lst, ratios=(0.7, 0.1, 0.2)):
    random.shuffle(lst)
    n = len(lst)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

train_c, val_c, test_c = split(has_change)
train_n, val_n, test_n = split(no_change)

train_files = sorted(train_c + train_n)
val_files = sorted(val_c + val_n)
test_files = sorted(test_c + test_n)

print(f"train: {len(train_files)} | val: {len(val_files)} | test: {len(test_files)}")

# 先清理 train 目录中不属于 train 的文件（因为源和目标重叠）
val_test_set = set(val_files + test_files)
for sub in ["A", "B", "label"]:
    sub_dir = os.path.join(base, "train", sub)
    for f in os.listdir(sub_dir):
        if f in val_test_set:
            os.remove(os.path.join(sub_dir, f))
# 也清理 train/list
train_list_dir = os.path.join(base, "train", "list")
if os.path.exists(train_list_dir):
    for f in os.listdir(train_list_dir):
        os.remove(os.path.join(train_list_dir, f))

# 创建目标目录并复制文件
for split_name, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
    for sub in ["A", "B", "label"]:
        os.makedirs(os.path.join(base, split_name, sub), exist_ok=True)

    for f in file_list:
        # A 和 B 文件名格式：年份_序号.png
        # label 和 A/B 同名
        src_a = os.path.join(src_A, f)
        src_b = os.path.join(src_B, f)
        src_l = os.path.join(src_label, f)

        dst_a = os.path.join(base, split_name, "A", f)
        dst_b = os.path.join(base, split_name, "B", f)
        dst_l = os.path.join(base, split_name, "label", f)

        if os.path.exists(src_a) and src_a != dst_a:
            shutil.copy2(src_a, dst_a)
        if os.path.exists(src_b) and src_b != dst_b:
            shutil.copy2(src_b, dst_b)
        if os.path.exists(src_l) and src_l != dst_l:
            shutil.copy2(src_l, dst_l)

    # 生成 list 文件（ChangeFormer 需要的格式）
    list_dir = os.path.join(base, split_name, "list")
    os.makedirs(list_dir, exist_ok=True)
    with open(os.path.join(list_dir, "train.txt" if split_name == "train" else "val.txt" if split_name == "val" else "test.txt"), "w") as fw:
        for f in file_list:
            fw.write(f"{f}\n")

print("划分完成！")

# 验证
for split_name in ["train", "val", "test"]:
    a_count = len(os.listdir(os.path.join(base, split_name, "A")))
    b_count = len(os.listdir(os.path.join(base, split_name, "B")))
    l_count = len(os.listdir(os.path.join(base, split_name, "label")))
    print(f"{split_name}: A={a_count}, B={b_count}, label={l_count}")
