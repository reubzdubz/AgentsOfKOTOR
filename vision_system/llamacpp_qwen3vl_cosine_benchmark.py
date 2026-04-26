#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import base64
import numpy as np
from openai import OpenAI

STATE_LABELS = ["combat", "narrative", "leveling"]
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


class LlamaCppVisionCosineBenchmark:
    def __init__(
        self,
        dataset_dir: str,
        base_url: str,
        model: str,
        api_key: str = "sk-no-key-required",
        output_dir: str = "output/llamacpp_qwen3vl_cosine_benchmark",
        random_state: int = 42,
        train_ratio: float = 0.8,
    ):
        self.dataset_path = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.train_ratio = train_ratio
        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
        self.model = model

    def load_dataset(self) -> List[Tuple[Path, str]]:
        items = []
        for state in STATE_LABELS:
            state_dir = self.dataset_path / state
            if not state_dir.exists():
                continue
            for pattern in IMAGE_EXTS:
                for img_path in sorted(state_dir.glob(pattern)):
                    items.append((img_path, state))
        return items

    def image_to_data_url(self, image_path: Path) -> str:
        import base64
        suffix = image_path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{b64}"


    def fetch_embedding(self, image_path: Path) -> np.ndarray:
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")

        resp = self.client.embeddings.create(
            model=self.model,
            input=[
                "Classify the state of the KOTOR UI in this image as one of: combat, narrative, or leveling."
            ],
            encoding_format="float",
        )

        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def extract_embeddings(self, items: List[Tuple[Path, str]]):
        embeddings = []
        labels = []
        rows = []
        latencies = []

        for image_path, label in items:
            start = time.time()
            vec = self.fetch_embedding(image_path)
            elapsed = time.time() - start
            embeddings.append(vec)
            labels.append(label)
            latencies.append(elapsed)
            rows.append({
                "path": str(image_path),
                "label": label,
                "latency_sec": elapsed,
            })
            print(f"{image_path.name}: label={label}, embed_time={elapsed:.3f}s")

        return np.vstack(embeddings), np.array(labels), rows, latencies

    def stratified_split(self, labels: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        train_idx = []
        test_idx = []
        for state in STATE_LABELS:
            idx = np.where(labels == state)[0]
            idx = idx.copy()
            rng.shuffle(idx)
            split = max(1, int(len(idx) * self.train_ratio))
            split = min(split, len(idx) - 1) if len(idx) > 1 else len(idx)
            train_idx.extend(idx[:split].tolist())
            test_idx.extend(idx[split:].tolist())
        return np.array(train_idx), np.array(test_idx)

    def build_centroids(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        centroids = {}
        for state in STATE_LABELS:
            state_vecs = X[y == state]
            if len(state_vecs) == 0:
                continue
            centroid = state_vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[state] = centroid.astype(np.float32)
        return centroids

    def predict(self, vec: np.ndarray, centroids: Dict[str, np.ndarray]):
        best_state = None
        best_score = -1.0
        scores = {}
        for state, centroid in centroids.items():
            score = float(np.dot(vec, centroid))
            scores[state] = score
            if score > best_score:
                best_score = score
                best_state = state
        return best_state, best_score, scores

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        metrics = {"overall_accuracy": float((y_true == y_pred).mean()), "per_state": {}}
        confusion = []
        for true_state in STATE_LABELS:
            row = []
            for pred_state in STATE_LABELS:
                row.append(int(np.sum((y_true == true_state) & (y_pred == pred_state))))
            confusion.append(row)

        for state in STATE_LABELS:
            tp = int(np.sum((y_pred == state) & (y_true == state)))
            fp = int(np.sum((y_pred == state) & (y_true != state)))
            fn = int(np.sum((y_pred != state) & (y_true == state)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            metrics["per_state"][state] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        metrics["confusion_matrix"] = {"labels": STATE_LABELS, "matrix": confusion}
        return metrics

    def benchmark(self):
        items = self.load_dataset()
        if not items:
            raise RuntimeError(f"No images found under {self.dataset_path}")

        X, y, rows, latencies = self.extract_embeddings(items)
        train_idx, test_idx = self.stratified_split(y)
        centroids = self.build_centroids(X[train_idx], y[train_idx])

        predictions = []
        prediction_rows = []
        for idx in test_idx:
            pred, score, scores = self.predict(X[idx], centroids)
            predictions.append(pred)
            prediction_rows.append({
                "path": rows[idx]["path"],
                "label": y[idx],
                "predicted": pred,
                "best_score": score,
                "scores": scores,
            })

        y_true = y[test_idx]
        y_pred = np.array(predictions)
        metrics = self.compute_metrics(y_true, y_pred)
        metrics.update({
            "total_images": int(len(y)),
            "train_images": int(len(train_idx)),
            "test_images": int(len(test_idx)),
            "average_embedding_latency_sec": float(sum(latencies) / len(latencies)),
            "train_ratio": float(self.train_ratio),
        })

        np.save(self.output_dir / "embeddings.npy", X)
        np.save(self.output_dir / "labels.npy", y)
        np.save(self.output_dir / "train_indices.npy", train_idx)
        np.save(self.output_dir / "test_indices.npy", test_idx)
        np.savez(self.output_dir / "centroids.npz", **centroids)

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(self.output_dir / "embedding_rows.json", "w") as f:
            json.dump(rows, f, indent=2)
        with open(self.output_dir / "predictions.json", "w") as f:
            json.dump(prediction_rows, f, indent=2)

        print("\n=== llama.cpp Vision Cosine Benchmark ===")
        print(f"Total images: {metrics['total_images']}")
        print(f"Train images: {metrics['train_images']}")
        print(f"Test images: {metrics['test_images']}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"Average embedding latency: {metrics['average_embedding_latency_sec']:.3f}s/image")
        print("\nPer-state metrics:")
        for state, stats in metrics["per_state"].items():
            print(f"  {state:10} | P: {stats['precision']:.2%} | R: {stats['recall']:.2%} | F1: {stats['f1']:.2%}")
        print("\nConfusion matrix:")
        print("labels:", ", ".join(metrics["confusion_matrix"]["labels"]))
        for row in metrics["confusion_matrix"]["matrix"]:
            print(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark cosine-similarity image embeddings from a llama.cpp OpenAI-compatible endpoint for KOTOR UI classification."
    )
    parser.add_argument("--dataset", default="vision_system/datasets/kotor_ui_samples")
    parser.add_argument("--base-url", default="http://0.0.0.0:8080/v1")
    parser.add_argument("--model", default="qwen3-vl-2b")
    parser.add_argument("--api-key", default="sk-no-key-required")
    parser.add_argument("--output-dir", default="output/llamacpp_qwen3vl_cosine_benchmark")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    runner = LlamaCppVisionCosineBenchmark(
        dataset_dir=args.dataset,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        output_dir=args.output_dir,
        random_state=args.random_state,
        train_ratio=args.train_ratio,
    )
    runner.benchmark()


if __name__ == "__main__":
    main()
