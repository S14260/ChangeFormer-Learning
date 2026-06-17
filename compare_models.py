"""
生成 LEVIR vs DSIFN 微调模型推理结果对比图（上下布局）
每张图展示两行：
  第一行：前期A | 后期B | GT真值
  第二行：LEVIR预测 | DSIFN预测 | 叠加图
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

base = os.path.dirname(os.path.abspath(__file__))
test_A = os.path.join(base, "datasets", "tianjin_wayback", "test", "A")
test_B = os.path.join(base, "datasets", "tianjin_wayback", "test", "B")
test_label = os.path.join(base, "datasets", "tianjin_wayback", "test", "label")
pred_LEVIR = os.path.join(base, "output_finetune_LEVIR_tianjinWayback")
pred_DSIFN = os.path.join(base, "output_finetune_DSIFN_tianjinWayback")
output_dir = os.path.join(base, "output_compare_levir_vs_dsifn")
os.makedirs(output_dir, exist_ok=True)

img_size = 256
margin = 6
label_height = 32

# 中文字体
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

def add_label(img, text):
    labeled = Image.new("RGB", (img.size[0], img.size[1] + label_height), (50, 50, 50))
    labeled.paste(img, (0, label_height))
    draw = ImageDraw.Draw(labeled)
    font = ImageFont.truetype(FONT_PATH, 18)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.size[0] - tw) // 2, 6), text, fill=(255, 255, 255), font=font)
    return labeled

# 读取文件列表
pred_files = sorted([f for f in os.listdir(pred_LEVIR) if f.endswith(".png")])
print(f"共 {len(pred_files)} 张对比图")

for fname in pred_files:
    img_A = Image.open(os.path.join(test_A, fname)).convert("RGB")
    img_B = Image.open(os.path.join(test_B, fname)).convert("RGB")
    gt = np.array(Image.open(os.path.join(test_label, fname)))
    pred_l = np.array(Image.open(os.path.join(pred_LEVIR, fname)))
    pred_d = np.array(Image.open(os.path.join(pred_DSIFN, fname)))

    # GT 着色：绿色
    gt_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    gt_rgb[gt == 255] = [0, 255, 0]

    # LEVIR 预测着色：红色
    levir_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    levir_rgb[pred_l == 255] = [255, 0, 0]

    # DSIFN 预测着色：蓝色
    dsifn_rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    dsifn_rgb[pred_d == 255] = [0, 100, 255]

    # 叠加图：A底图 + GT绿色 + LEVIR红色 + DSIFN蓝色
    overlay = np.array(img_A.copy(), dtype=np.float32)
    overlay[gt == 255] = overlay[gt == 255] * 0.5 + np.array([0, 255, 0]) * 0.5
    overlay[pred_l == 255] = overlay[pred_l == 255] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay[pred_d == 255] = overlay[pred_d == 255] * 0.5 + np.array([0, 100, 255]) * 0.5
    overlay = overlay.astype(np.uint8)

    # 加标签
    row1 = [add_label(img_A, "前期A"), add_label(img_B, "后期B"), add_label(Image.fromarray(gt_rgb), "GT真值(绿)")]
    row2 = [add_label(Image.fromarray(levir_rgb), "LEVIR预测(红)"), add_label(Image.fromarray(dsifn_rgb), "DSIFN预测(蓝)"), add_label(Image.fromarray(overlay), "叠加对比")]

    # 计算画布尺寸
    col_w = img_size + margin
    row_h = img_size + label_height + margin
    canvas_w = col_w * 3 - margin
    canvas_h = row_h * 2 - margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    # 贴第一行
    for i, img in enumerate(row1):
        canvas.paste(img, (i * col_w, 0))
    # 贴第二行
    for i, img in enumerate(row2):
        canvas.paste(img, (i * col_w, row_h))

    canvas.save(os.path.join(output_dir, fname))
    print(f"  {fname}")

print(f"\n对比图保存在: {output_dir}")
