#!/usr/bin/env python3
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


STATE_LABELS = ["combat", "narrative", "leveling"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DATASET_DIR = "vision_system/datasets/kotor_ui_samples"
OUTPUT_DIR = "output/qwen3vl_label_similarity_benchmark"

LABEL_TEXTS = {
    "combat": "A game UI screenshot showing combat gameplay, battle actions, hostile targets, health bars, loot pickup overlays, item collected popups, or post-combat reward screens.",
    "narrative": "A game UI screenshot showing narrative dialogue, story conversation, subtitles, character interaction, or cutscene text.",
    "leveling": "A game UI screenshot showing character leveling, progression, skills, attributes, inventory stats, or upgrade menus.",
}


def load_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    items = []
    for state in STATE_LABELS:
        state_dir = dataset_path / state
        if not state_dir.exists():
            continue
        for ext in IMAGE_EXTS:
            for img_path in sorted(state_dir.glob(f"*{ext}")):
                items.append((img_path, state))
    return items


def build_label_embeddings(model):
    labels = list(LABEL_TEXTS.keys())
    texts = [LABEL_TEXTS[k] for k in labels]
    embs = model.encode(texts, normalize_embeddings=True)
    return labels, np.asarray(embs, dtype=np.float32)


def fetch_image_embedding(model, image_path):
    emb = model.encode([str(image_path)], normalize_embeddings=True)
    return np.asarray(emb[0], dtype=np.float32)


def predict_image(image_emb, labels, label_embs):
    scores = label_embs @ image_emb
    best_idx = int(np.argmax(scores))
    return labels[best_idx], float(scores[best_idx]), {
        labels[i]: float(scores[i]) for i in range(len(labels))
    }


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "overall_accuracy": float((y_true == y_pred).mean())
    }

    confusion = []
    for true_state in STATE_LABELS:
        row = []
        for pred_state in STATE_LABELS:
            row.append(int(np.sum((y_true == true_state) & (y_pred == pred_state))))
        confusion.append(row)

    per_state = {}
    for state in STATE_LABELS:
        tp = int(np.sum((y_pred == state) & (y_true == state)))
        fp = int(np.sum((y_pred == state) & (y_true != state)))
        fn = int(np.sum((y_pred != state) & (y_true == state)))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_state[state] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    metrics["per_state"] = per_state
    metrics["confusion_matrix"] = {
        "labels": STATE_LABELS,
        "matrix": confusion,
    }
    return metrics


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
    )

    print(f"Loading dataset from: {DATASET_DIR}")
    items = load_dataset(DATASET_DIR)
    if not items:
        raise RuntimeError(f"No images found under {DATASET_DIR}")

    print(f"Loaded {len(items)} images")

    labels, label_embs = build_label_embeddings(model)
    print("Built label embeddings:")
    for label in labels:
        print(f"  - {label}: {LABEL_TEXTS[label]}")

    predictions = []
    y_true = []
    embedding_rows = []
    prediction_rows = []
    latencies = []

    for image_path, true_label in items:
        start = time.time()
        image_emb = fetch_image_embedding(model, image_path)
        pred_label, best_score, scores = predict_image(image_emb, labels, label_embs)
        elapsed = time.time() - start

        y_true.append(true_label)
        predictions.append(pred_label)
        latencies.append(elapsed)

        embedding_rows.append({
            "path": str(image_path),
            "label": true_label,
            "latency_sec": elapsed,
        })

        prediction_rows.append({
            "path": str(image_path),
            "label": true_label,
            "predicted": pred_label,
            "best_score": best_score,
            "scores": scores,
        })

        print(
            f"{image_path.name}: "
            f"label={true_label}, pred={pred_label}, "
            f"scores={{{', '.join(f'{k}: {v:.4f}' for k, v in scores.items())}}}, "
            f"time={elapsed:.3f}s"
        )

    metrics = compute_metrics(y_true, predictions)
    metrics.update({
        "total_images": int(len(items)),
        "average_embedding_latency_sec": float(sum(latencies) / len(latencies)),
        "model_name": EMBEDDING_MODEL_NAME,
        "label_texts": LABEL_TEXTS,
    })

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "embedding_rows.json", "w") as f:
        json.dump(embedding_rows, f, indent=2)

    with open(output_dir / "predictions.json", "w") as f:
        json.dump(prediction_rows, f, indent=2)

    print("\nQwen3-VL Label Similarity Benchmark")
    print(f"Total images: {metrics['total_images']}")
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Average embedding latency: {metrics['average_embedding_latency_sec']:.3f}s/image")

    print("\nPer-state metrics:")
    for state, stats in metrics["per_state"].items():
        print(
            f"  {state:10s} "
            f"P={stats['precision']:.2f} "
            f"R={stats['recall']:.2f} "
            f"F1={stats['f1']:.2f}"
        )

    print("\nConfusion matrix:")
    print("labels:", ", ".join(metrics["confusion_matrix"]["labels"]))
    for row in metrics["confusion_matrix"]["matrix"]:
        print(row)


if __name__ == "__main__":
    main()