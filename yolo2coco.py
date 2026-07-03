import json
import shutil
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path("yolox_tr")

yaml_file = ROOT / "data.yaml"

with open(yaml_file, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]

if isinstance(names, dict):
    categories = [
        {
            "id": int(i) + 1,
            "name": names[i],
            "supercategory": "object",
        }
        for i in sorted(names.keys())
    ]
else:
    categories = [
        {
            "id": i + 1,
            "name": name,
            "supercategory": "object",
        }
        for i, name in enumerate(names)
    ]


(ROOT / "annotations").mkdir(exist_ok=True)


def convert_split(split_name):

    image_dir = ROOT / split_name / "images"
    label_dir = ROOT / split_name / "labels"

    out_img_dir = ROOT / f"{split_name}2017"
    out_img_dir.mkdir(exist_ok=True)

    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    for image_path in sorted(image_dir.iterdir()):

        if image_path.suffix.lower() not in exts:
            continue

        shutil.copy2(image_path, out_img_dir / image_path.name)

        w, h = Image.open(image_path).size

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": w,
                "height": h,
            }
        )

        label_path = label_dir / (image_path.stem + ".txt")

        if label_path.exists():

            with open(label_path, "r") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    cls, xc, yc, bw, bh = map(float, line.split())

                    bw *= w
                    bh *= h

                    x = xc * w - bw / 2
                    y = yc * h - bh / 2

                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(cls) + 1,
                            "bbox": [
                                round(x, 2),
                                round(y, 2),
                                round(bw, 2),
                                round(bh, 2),
                            ],
                            "area": round(bw * bh, 2),
                            "iscrowd": 0,
                        }
                    )

                    annotation_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(
        ROOT / "annotations" / f"instances_{split_name}2017.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(coco, f, indent=4)


convert_split("train")
convert_split("val")

print("Done!")