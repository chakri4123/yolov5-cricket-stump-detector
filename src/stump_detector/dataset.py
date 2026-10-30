"""Turn a multi-class YOLO export into a single-class stump dataset."""
import shutil
from pathlib import Path

import yaml

SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def stump_class_ids(names: list[str] | dict[int, str]) -> set[int]:
    """Find stump classes by name. The source dataset spells the class 'Stamps', so match both spellings."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(index) for index, name in items if name.lower().startswith(("stump", "stamp"))}


def filter_label_lines(lines: list[str], keep_ids: set[int]) -> list[str]:
    """Keep only stump boxes and renumber them as class 0."""
    kept = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5 and int(parts[0]) in keep_ids:
            kept.append(" ".join(["0", *parts[1:5]]))
    return kept


def prepare_dataset(source: Path, output: Path) -> dict[str, dict[str, int]]:
    """Copy each split to output, keeping stump labels only.

    Images with no stump stay in the dataset with an empty label file, so the model also learns
    from frames that have no stumps in them.
    """
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    keep_ids = stump_class_ids(config["names"])
    if not keep_ids:
        raise ValueError(f"no stump class found in {config['names']}")

    stats: dict[str, dict[str, int]] = {}
    for source_split, split in SPLITS.items():
        images_dir = source / source_split / "images"
        if not images_dir.is_dir():
            continue
        out_images = output / "images" / split
        out_labels = output / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        counts = {"images": 0, "with_stumps": 0, "boxes": 0}
        for image in sorted(images_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = source / source_split / "labels" / f"{image.stem}.txt"
            lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
            kept = filter_label_lines(lines, keep_ids)
            shutil.copy2(image, out_images / image.name)
            (out_labels / f"{image.stem}.txt").write_text("\n".join(kept), encoding="utf-8")
            counts["images"] += 1
            counts["with_stumps"] += bool(kept)
            counts["boxes"] += len(kept)
        stats[split] = counts

    data = {"path": str(output.resolve()), "train": "images/train", "val": "images/val", "names": {0: "stumps"}}
    if "test" in stats:
        data["test"] = "images/test"
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return stats
