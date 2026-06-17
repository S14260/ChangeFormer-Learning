import json
import os
import numpy as np
from PIL import Image, ImageDraw

# 配置路径
json_dir = "train/A"  # labelme json 在 A 文件夹里
label_dir = "train/label"  # 输出到 label 文件夹
img_size = (256, 256)

os.makedirs(label_dir, exist_ok=True)

json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
print(f"找到 {len(json_files)} 个 JSON 文件")

for jf in sorted(json_files):
    json_path = os.path.join(json_dir, jf)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 创建黑底 mask
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        if shape["shape_type"] == "polygon":
            points = [(int(p[0]), int(p[1])) for p in shape["points"]]
            draw.polygon(points, fill=255)

    # 保存为 png，文件名和 A/B 图片一致
    png_name = jf.replace(".json", ".png")
    mask.save(os.path.join(label_dir, png_name))

print(f"转换完成，共生成 {len(json_files)} 张 label 到 {label_dir}/")
