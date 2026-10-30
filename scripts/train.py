"""Train YOLOv5s on the stump dataset using the settings in configs/train.yaml."""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "stumps" / "data.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    parser.add_argument("--epochs", type=int, help="Override the epochs in the config")
    parser.add_argument("--name", default="stumps")
    args = parser.parse_args()

    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    weights = settings.pop("model")
    if args.epochs:
        settings["epochs"] = args.epochs

    YOLO(weights).train(data=str(args.data), project=str(ROOT / "runs"), name=args.name, exist_ok=True, **settings)


if __name__ == "__main__":
    main()
