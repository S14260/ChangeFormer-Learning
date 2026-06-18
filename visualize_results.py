"""
可视化 ChangeFormer 推理结果
将前期影像(A)、后期影像(B)、预测结果(Pred) 拼接输出
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding='utf-8')

A_DIR = r"E:\changeformer\ChangeFormer\datasets\tianjin_wayback\test\A"
B_DIR = r"E:\changeformer\ChangeFormer\datasets\tianjin_wayback\test\B"
PRED_DIR = r"E:\changeformer\ChangeFormer\output_finetune_LEVIR_enhanced_tianjinWayback"
OUT_DIR = r"E:\changeformer\ChangeFormer\output_finetune_LEVIR_enhanced_tianjinWayback\vis"

IMG_SIZE = (256, 256)  # 展示尺寸


def add_label(img, text):
    """在图片顶部添加标签"""
    labeled = img.copy()
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, labeled.width, 24], fill=(0, 0, 0))
    draw.text((8, 4), text, fill=(255, 255, 255), font=font)
    return labeled


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(PRED_DIR) if f.endswith(".png") and not os.path.isdir(os.path.join(PRED_DIR, f))])
    print(f"共 {len(pred_files)} 个预测结果")

    for fname in pred_files:
        a_path = os.path.join(A_DIR, fname)
        b_path = os.path.join(B_DIR, fname)
        p_path = os.path.join(PRED_DIR, fname)

        if not os.path.exists(a_path) or not os.path.exists(b_path):
            continue

        img_a = Image.open(a_path).convert("RGB").resize(IMG_SIZE)
        img_b = Image.open(b_path).convert("RGB").resize(IMG_SIZE)
        img_p = Image.open(p_path).convert("L").resize(IMG_SIZE)

        # 预测结果转为红色叠加
        pred_arr = np.array(img_p)
        overlay = np.array(img_b).copy()
        overlay[pred_arr > 128] = [255, 0, 0]  # 红色标记变化区域
        img_overlay = Image.fromarray(overlay)

        # 标注
        a_labeled = add_label(img_a, "前期 (A)")
        b_labeled = add_label(img_b, "后期 (B)")
        pred_labeled = add_label(img_p.convert("RGB"), "预测 (Pred)")
        overlay_labeled = add_label(img_overlay, "叠加 (Overlay)")

        # 拼接: A | B | Pred | Overlay
        gap = 4
        total_w = IMG_SIZE[0] * 4 + gap * 3
        canvas = Image.new("RGB", (total_w, IMG_SIZE[1] + 24), (255, 255, 255))
        canvas.paste(a_labeled, (0, 0))
        canvas.paste(b_labeled, (IMG_SIZE[0] + gap, 0))
        canvas.paste(pred_labeled, (IMG_SIZE[0] * 2 + gap * 2, 0))
        canvas.paste(overlay_labeled, (IMG_SIZE[0] * 3 + gap * 3, 0))

        out_path = os.path.join(OUT_DIR, fname)
        canvas.save(out_path)

    print(f"可视化结果保存在: {OUT_DIR}")
    print(f"共生成 {len(os.listdir(OUT_DIR))} 张对比图")

if __name__ == "__main__":
    main()
