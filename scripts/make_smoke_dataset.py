"""Draw a tiny fake dataset, to test the pipeline without the real data.

Models trained on this are only a pipeline check. Their numbers mean nothing about real stumps.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def draw_sample(rng: random.Random, size: int = 320) -> tuple[np.ndarray, list[str]]:
    image = np.full((size, size, 3), (60, 140, 70), dtype=np.uint8)
    width, height = rng.randint(24, 50), rng.randint(70, 130)
    x, y = rng.randint(0, size - width), rng.randint(0, size - height)
    for offset in (0, width // 2, width - 4):
        cv2.rectangle(image, (x + offset, y), (x + offset + 3, y + height), (210, 225, 235), -1)
    label = f"0 {(x + width / 2) / size:.6f} {(y + height / 2) / size:.6f} {width / size:.6f} {height / size:.6f}"
    return image, [label]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "smoke")
    args = parser.parse_args()

    rng = random.Random(42)
    for split, count in (("train", 24), ("val", 8)):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image, labels = draw_sample(rng)
            cv2.imwrite(str(args.output / "images" / split / f"{index:03d}.jpg"), image)
            (args.output / "labels" / split / f"{index:03d}.txt").write_text("\n".join(labels), encoding="utf-8")

    data = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "stumps"}}
    (args.output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Wrote {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
