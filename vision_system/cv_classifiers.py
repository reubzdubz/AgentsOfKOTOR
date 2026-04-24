"""
Naive computer vision classifiers for KOTOR state detection.
Uses histograms, edge detection, and template matching.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from enum import Enum


class GameState(Enum):
    """Game state enum."""
    COMBAT = "combat"
    NARRATIVE = "narrative"
    LEVELING = "leveling"
    UNKNOWN = "unknown"


class HistogramClassifier:
    """
    Classifies based on color histogram distribution.
    Assumes combat/narrative/leveling have distinct color schemes.
    """
    
    def __init__(self):
        self.templates = {}  # state -> histogram
        self.trained = False
    
    def train(self, dataset_dir: str):
        """Train on dataset by computing average histogram per state."""
        dataset_path = Path(dataset_dir)
        
        for state in ["combat", "narrative", "leveling"]:
            state_dir = dataset_path / state
            hists = []
            
            for img_path in state_dir.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Compute histogram for each channel
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], 
                                   [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hists.append(hist)
            
            if hists:
                # Store average histogram for this state
                self.templates[state] = np.mean(hists, axis=0)
        
        self.trained = len(self.templates) > 0
    
    def predict(self, frame: np.ndarray) -> Tuple[GameState, float]:
        """
        Classify frame based on histogram similarity.
        
        Returns:
            (GameState, confidence)
        """
        if not self.trained:
            return GameState.UNKNOWN, 0.0
        
        # Compute histogram of input frame
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Compare against all templates
        best_match = GameState.UNKNOWN
        best_score = -1
        
        for state, template_hist in self.templates.items():
            # Use correlation as similarity metric
            score = cv2.compareHist(hist, template_hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match = GameState(state)
        
        return best_match, best_score


class EdgeClassifier:
    """
    Classifies based on edge density.
    Combat UI has more edges (buttons, bars), narrative is cleaner.
    """
    
    def __init__(self, threshold: float = 100.0):
        self.threshold = threshold
        self.stats = {}
    
    def train(self, dataset_dir: str):
        """Compute edge statistics per state."""
        dataset_path = Path(dataset_dir)
        
        for state in ["combat", "narrative", "leveling"]:
            state_dir = dataset_path / state
            edge_densities = []
            
            for img_path in state_dir.glob("*.png"):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                density = np.sum(edges > 0) / edges.size
                edge_densities.append(density)
            
            if edge_densities:
                self.stats[state] = {
                    "mean": np.mean(edge_densities),
                    "std": np.std(edge_densities)
                }
    
    def predict(self, frame: np.ndarray) -> Tuple[GameState, float]:
        """
        Classify based on edge density.
        
        Returns:
            (GameState, confidence)
        """
        if not self.stats:
            return GameState.UNKNOWN, 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = np.sum(edges > 0) / edges.size
        
        # Find closest mean
        best_match = GameState.UNKNOWN
        best_distance = float('inf')
        
        for state, stats in self.stats.items():
            distance = abs(density - stats["mean"])
            if distance < best_distance:
                best_distance = distance
                best_match = GameState(state)
        
        # Confidence inversely related to distance
        confidence = 1.0 - (best_distance / max(s["mean"] for s in self.stats.values()))
        confidence = max(0.0, min(1.0, confidence))
        
        return best_match, confidence


class EmbeddingClassifier:
    """
    Classifies using image embeddings from a CLIP-style model.
    This is a good compromise between simple CV heuristics and a full VLM.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self.state_templates = {}

    def _ensure_model_loaded(self):
        if self.model is None or self.processor is None:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            self.torch = torch
            self.Image = Image
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model.eval()

    def _encode_image(self, image: np.ndarray):
        self._ensure_model_loaded()
        image = self.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().reshape(-1)

    def train(self, dataset_dir: str):
        dataset_path = Path(dataset_dir)

        for state in ["combat", "narrative", "leveling"]:
            state_dir = dataset_path / state
            embeddings = []

            for img_path in state_dir.glob("*.png"):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                embedding = self._encode_image(frame)
                embeddings.append(embedding)

            if embeddings:
                self.state_templates[state] = np.mean(embeddings, axis=0)

    def predict(self, frame: np.ndarray) -> Tuple[GameState, float]:
        if not self.state_templates:
            return GameState.UNKNOWN, 0.0

        frame_embedding = self._encode_image(frame)

        best_match = GameState.UNKNOWN
        best_score = -1.0
        for state, template_embedding in self.state_templates.items():
            score = float(np.dot(frame_embedding, template_embedding) / (
                np.linalg.norm(frame_embedding) * np.linalg.norm(template_embedding)
            ))
            if score > best_score:
                best_score = score
                best_match = GameState(state)

        return best_match, max(0.0, min(1.0, best_score))


class TemplateDetector:
    """
    Detects specific UI elements via template matching.
    Look for combat bars, health indicators, dialogue boxes.
    """
    
    def __init__(self):
        self.templates = {}  # name -> template image
    
    def add_template(self, name: str, template_path: str):
        """Register a template to look for."""
        template = cv2.imread(template_path)
        if template is not None:
            self.templates[name] = template
    
    def detect(self, frame: np.ndarray, threshold: float = 0.7) -> Dict[str, List]:
        """
        Detect all registered templates in frame.
        
        Returns:
            Dict mapping template names to list of (x, y, confidence) detections.
        """
        detections = {}
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        for name, template in self.templates.items():
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale matching
            matches = []
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5]:
                w = int(gray_template.shape[1] * scale)
                h = int(gray_template.shape[0] * scale)
                
                if w > gray_frame.shape[1] or h > gray_frame.shape[0]:
                    continue
                
                resized = cv2.resize(gray_template, (w, h))
                result = cv2.matchTemplate(gray_frame, resized, cv2.TM_CCOEFF)
                
                # Find peaks
                y, x = np.unravel_index(np.argmax(result), result.shape)
                confidence = result[y, x]
                
                if confidence > threshold:
                    matches.append((x, y, confidence))
            
            detections[name] = matches
        
        return detections


class EnsembleClassifier:
    """
    Combines multiple classifiers via voting.
    """
    
    def __init__(self, use_embedding: bool = False, embedding_model_name: str = "openai/clip-vit-base-patch32"):
        self.histogram = HistogramClassifier()
        self.edge = EdgeClassifier()
        self.classifiers = [self.histogram, self.edge]

        self.embedding = None
        if use_embedding:
            self.embedding = EmbeddingClassifier(model_name=embedding_model_name)
            self.classifiers.append(self.embedding)
    
    def train(self, dataset_dir: str):
        """Train all sub-classifiers."""
        for clf in self.classifiers:
            clf.train(dataset_dir)
    
    def predict(self, frame: np.ndarray) -> Tuple[GameState, float]:
        """
        Ensemble prediction with averaging.
        
        Returns:
            (GameState, average_confidence)
        """
        votes = {}
        confidences = {}
        
        for clf in self.classifiers:
            state, conf = clf.predict(frame)
            if state != GameState.UNKNOWN:
                votes[state.value] = votes.get(state.value, 0) + 1
                confidences[state.value] = confidences.get(state.value, 0) + conf
        
        if not votes:
            return GameState.UNKNOWN, 0.0
        
        # Winner by vote count
        best_state = max(votes, key=votes.get)
        avg_confidence = confidences[best_state] / votes[best_state]
        
        return GameState(best_state), avg_confidence
