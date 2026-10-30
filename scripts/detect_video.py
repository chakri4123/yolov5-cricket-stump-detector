"""Run the trained detector on a video file, or a webcam with --source 0, and report the real speed."""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to a video, or a webcam index such as 0")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs" / "stumps" / "weights" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "output.mp4")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--show", action="store_true", help="Show a live window. Press q to stop.")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    frames = detections = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=args.conf, verbose=False)[0]
            detections += len(result.boxes)
            frames += 1
            annotated = result.plot()
            writer.write(annotated)
            if args.show:
                cv2.imshow("Stump detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"{frames} frames, {detections} detections, {frames / elapsed:.1f} FPS. Saved to {args.output}")


if __name__ == "__main__":
    main()
