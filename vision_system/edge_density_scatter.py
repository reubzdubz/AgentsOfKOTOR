#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATE_LABELS = ["combat", "narrative", "leveling"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def find_images(dataset_dir: Path):
    rows = []
    for label in STATE_LABELS:
        class_dir = dataset_dir / label
        if not class_dir.exists():
            continue
        for ext in IMAGE_EXTS:
            for image_path in sorted(class_dir.glob(f"*{ext}")):
                rows.append({"label": label, "path": str(image_path)})
    return rows


def edge_metrics(image_path: Path, low_thresh: int, high_thresh: int, blur_ksize: int):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, low_thresh, high_thresh)

    h, w = edges.shape
    x_density = edges.sum(axis=0).astype(np.float32) / 255.0 / h
    y_density = edges.sum(axis=1).astype(np.float32) / 255.0 / w

    return {
        "edge_density_x_mean": float(x_density.mean()),
        "edge_density_y_mean": float(y_density.mean()),
        "edge_density_x_std": float(x_density.std()),
        "edge_density_y_std": float(y_density.std()),
        "edge_density_global": float((edges > 0).mean()),
        "width": int(w),
        "height": int(h),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute edge densities by class and scatter plot x/y edge density with matplotlib.")
    parser.add_argument("--dataset", default="vision_system/datasets/kotor_ui_samples")
    parser.add_argument("--output-dir", default="output/edge_density_scatter_matplotlib")
    parser.add_argument("--low-thresh", type=int, default=100)
    parser.add_argument("--high-thresh", type=int, default=200)
    parser.add_argument("--blur-ksize", type=int, default=3)
    parser.add_argument("--figsize", type=float, nargs=2, default=[9, 7])
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--point-size", type=float, default=28)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = find_images(dataset_dir)
    if not rows:
        raise RuntimeError(f"No images found under {dataset_dir}")

    out_rows = []
    for row in rows:
        metrics = edge_metrics(Path(row["path"]), args.low_thresh, args.high_thresh, args.blur_ksize)
        out_rows.append({**row, **metrics})

    df = pd.DataFrame(out_rows)
    csv_path = output_dir / "edge_density_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary = df.groupby("label")[["edge_density_x_mean", "edge_density_y_mean", "edge_density_global"]].agg(["mean", "std", "count"])
    summary_path = output_dir / "edge_density_summary.csv"
    summary.to_csv(summary_path)

    color_map = {
        "combat": "tab:red",
        "narrative": "tab:blue",
        "leveling": "tab:green",
    }

    plt.figure(figsize=tuple(args.figsize))
    for label in STATE_LABELS:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["edge_density_x_mean"],
            sub["edge_density_y_mean"],
            s=args.point_size,
            alpha=args.alpha,
            label=label,
            c=color_map.get(label, None),
        )

    plt.title(f"Edge Density by Class (Canny {args.low_thresh}-{args.high_thresh})")
    plt.xlabel("X edge density")
    plt.ylabel("Y edge density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    chart_path = output_dir / "edge_density_scatter.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()