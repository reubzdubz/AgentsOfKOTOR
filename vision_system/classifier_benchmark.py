#!/usr/bin/env python3
"""
Benchmark classifiers on collected dataset.
Provides accuracy, precision, recall metrics.

Usage:
    python classifier_benchmark.py --dataset datasets/kotor_ui_samples
"""

import cv2
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from cv_classifiers import (
    HistogramClassifier,
    EdgeClassifier,
    EnsembleClassifier,
    GameState
)


class Benchmark:
    def __init__(self, dataset_dir: str):
        self.dataset_path = Path(dataset_dir)
        self.results = {}
    
    def load_test_set(self) -> Tuple[List[Tuple[str, str]], int]:
        """
        Load test images from dataset.
        
        Returns:
            List of (image_path, true_state), total_images
        """
        images = []
        for state in ["combat", "narrative", "leveling"]:
            state_dir = self.dataset_path / state
            for img_path in sorted(state_dir.glob("*.png")):
                images.append((str(img_path), state))
        
        return images, len(images)
    
    def evaluate(self, classifier, images: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate classifier on test set.
        
        Returns:
            Dict with accuracy, precision, recall per state
        """
        predictions = []
        ground_truth = []
        
        for img_path, true_state in images:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            
            pred_state, conf = classifier.predict(frame)
            predictions.append(pred_state.value)
            ground_truth.append(true_state)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, ground_truth)
        return metrics
    
    def _compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        """Compute accuracy, precision, recall, F1."""
        states = ["combat", "narrative", "leveling"]
        metrics = {"overall_accuracy": 0, "per_state": {}}
        
        # Overall accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        metrics["overall_accuracy"] = correct / len(ground_truth) if ground_truth else 0
        
        # Per-state metrics
        for state in states:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == state and g == state)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == state and g != state)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != state and g == state)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics["per_state"][state] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        
        return metrics
    
    def run(self):
        """Run full benchmark."""
        images, total = self.load_test_set()
        
        if total == 0:
            print("No images found in dataset. Collect samples first!")
            return
        
        print(f"\n=== KOTOR Classifier Benchmark ===")
        print(f"Dataset: {total} images\n")
        
        classifiers = {
            "Histogram": HistogramClassifier(),
            "Edge Density": EdgeClassifier(),
            "Ensemble": EnsembleClassifier()
        }
        
        # Train all classifiers
        print("Training classifiers...")
        for name, clf in classifiers.items():
            clf.train(str(self.dataset_path))
        
        # Evaluate each
        print("\nEvaluating classifiers...\n")
        for name, clf in classifiers.items():
            metrics = self.evaluate(clf, images)
            self.results[name] = metrics
            self._print_metrics(name, metrics)
    
    def _print_metrics(self, name: str, metrics: Dict):
        """Pretty-print metrics."""
        print(f"\n{name}:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"  Per-State Performance:")
        
        for state, stats in metrics["per_state"].items():
            print(f"    {state:10} | P: {stats['precision']:.2%} | R: {stats['recall']:.2%} | F1: {stats['f1']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark classifiers")
    parser.add_argument("--dataset", default="datasets/kotor_ui_samples",
                        help="Path to dataset directory")
    args = parser.parse_args()
    
    benchmark = Benchmark(args.dataset)
    benchmark.run()


if __name__ == "__main__":
    main()
