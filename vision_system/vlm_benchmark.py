#!/usr/bin/env python3
import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TextStreamer,
)

STATE_LABELS = ["combat", "narrative", "leveling"]
STATE_ALIASES = {
    "battle": "combat",
    "fight": "combat",
    "enemy": "combat",
    "dialogue": "narrative",
    "conversation": "narrative",
    "talk": "narrative",
    "level up": "leveling",
    "leveling": "leveling",
    "upgrade": "leveling",
    "skills": "leveling",
}

PROMPT_TEMPLATE = (
    "Classify the following KOTOR user interface screenshot into exactly one of these categories: "
    "combat, narrative, or leveling. Only answer with one word: combat, narrative, or leveling."
)


class VlmBenchmark:
    def __init__(self, dataset_dir: str, model_id: str, peft_path: str = None):
        self.dataset_path = Path(dataset_dir)
        self.model_id = model_id
        self.peft_path = peft_path
        self.processor = None
        self.model = None
        self.streamer = None

    def load_model(self):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=[
                "model.vision_backbone",
                "model.transformer.ff_out",
                "model.transformer.ln_f",
            ],
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
            token=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=nf4_config,
            token=True,
        )

        if self.peft_path:
            self.model = PeftModel.from_pretrained(self.model, self.peft_path)

        self.streamer = TextStreamer(tokenizer=self.processor.tokenizer, skip_special_tokens=True)

    def load_dataset(self) -> List[Tuple[str, str]]:
        images = []
        for state in STATE_LABELS:
            state_dir = self.dataset_path / state
            if not state_dir.exists():
                continue
            for img_path in sorted(state_dir.glob("*.png")):
                images.append((str(img_path), state))
        return images

    def normalize_prediction(self, text: str) -> str:
        if not text:
            return "unknown"

        normalized = text.lower().strip()
        normalized = re.sub(r"[^a-z ]+", " ", normalized)

        for state in STATE_LABELS:
            if state in normalized:
                return state

        for alias, label in STATE_ALIASES.items():
            if alias in normalized:
                return label

        tokens = normalized.split()
        for token in tokens:
            if token in STATE_LABELS:
                return token

        return "unknown"

    def classify_image(self, image_path: str, prompt: str) -> Tuple[str, float]:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        kwargs = {
            "max_new_tokens": 64,
            "streamer": self.streamer,
        }

        if self.model.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                generated_ids = self.model.generate(**inputs, **kwargs)
        else:
            generated_ids = self.model.generate(**inputs, **kwargs)

        generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        predicted_state = self.normalize_prediction(generated_text)
        confidence = 1.0 if predicted_state != "unknown" else 0.0
        return predicted_state, confidence

    def compute_metrics(self, predictions: List[str], ground_truth: List[str]) -> Dict:
        metrics = {"overall_accuracy": 0.0, "per_state": {}}
        total = len(ground_truth)
        if total == 0:
            return metrics

        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        metrics["overall_accuracy"] = correct / total

        for state in STATE_LABELS:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == state and g == state)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == state and g != state)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != state and g == state)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            metrics["per_state"][state] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        return metrics

    def run(self, prompt: str):
        self.load_model()
        images = self.load_dataset()
        total = len(images)
        if total == 0:
            raise RuntimeError(
                f"No images found under {self.dataset_path}. "
                "Ensure the dataset has combat, narrative, and leveling subfolders."
            )

        predictions = []
        ground_truth = []
        timings = []

        for img_path, true_state in images:
            start = time.time()
            predicted_state, _ = self.classify_image(img_path, prompt)
            elapsed = time.time() - start
            timings.append(elapsed)

            predictions.append(predicted_state)
            ground_truth.append(true_state)
            print(f"{Path(img_path).name}: expected={true_state}, predicted={predicted_state}, time={elapsed:.2f}s")

        metrics = self.compute_metrics(predictions, ground_truth)
        self.print_summary(metrics, total, timings)

    @staticmethod
    def print_summary(metrics: Dict, total: int, timings: List[float]):
        print("\n=== VLM UI Classification Benchmark ===")
        print(f"Total images: {total}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"Average latency: {sum(timings)/len(timings):.2f}s/image")
        print("\nPer-state metrics:")
        for state, stats in metrics["per_state"].items():
            print(
                f"  {state:10} | P: {stats['precision']:.2%} | R: {stats['recall']:.2%} | F1: {stats['f1']:.2%}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a VLM on KOTOR UI state images.")
    parser.add_argument(
        "--dataset",
        default="vision_system/datasets/kotor_ui_samples",
        help="Path to the labeled UI dataset directory.",
    )
    parser.add_argument(
        "--model",
        default="../Molmo2-VLA/Molmo2-4B",
        help="Transformers model path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--peft",
        default="checkpoint-8100",
        help="Optional PEFT checkpoint folder for the model.",
    )
    parser.add_argument(
        "--prompt",
        default=PROMPT_TEMPLATE,
        help="Prompt template to send to the VLM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = VlmBenchmark(dataset_dir=args.dataset, model_id=args.model, peft_path=args.peft)
    benchmark.run(prompt=args.prompt)


if __name__ == "__main__":
    main()
