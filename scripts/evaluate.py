"""Measure a trained model on a dataset split and save the metrics to results/metrics.json."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "runs" / "stumps" / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "stumps" / "data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "metrics.json")
    args = parser.parse_args()

    metrics = YOLO(str(args.weights)).val(data=str(args.data), split=args.split, project=str(ROOT / "runs"), name=f"eval_{args.split}", exist_ok=True)
    results = {
        "split": args.split,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "inference_ms_per_image": round(float(metrics.speed["inference"]), 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
