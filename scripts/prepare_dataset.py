"""Keep only the stump class from the raw download."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stump_detector.dataset import prepare_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "stumps")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
