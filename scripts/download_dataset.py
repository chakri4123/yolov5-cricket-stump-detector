"""Download the source dataset from Roboflow Universe. Needs ROBOFLOW_API_KEY in .env."""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key.")

    project = Roboflow(api_key=key).workspace("lbwdrs").project("bat-ball-stumps-detection-uoehl")
    project.version(args.version).download("yolov8", location=str(args.output), overwrite=True)
    print(f"Downloaded to {args.output}")


if __name__ == "__main__":
    main()
