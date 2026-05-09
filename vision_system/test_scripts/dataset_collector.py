#!/usr/bin/env python3
"""
Dataset collector for KOTOR UI samples.
Captures screenshots and allows manual labeling for state classification.

Usage:
    python dataset_collector.py --mode collect
    python dataset_collector.py --mode label
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import mss
import cv2
import numpy as np
import winsound


class DatasetCollector:
    def __init__(self, base_dir: str = "datasets/kotor_ui_samples"):
        self.base_dir = Path(base_dir)
        self.manifest_path = self.base_dir / "manifest.json"
        self.states = ["combat", "narrative", "leveling"]
        
        # Create directories if they don't exist
        for state in self.states:
            (self.base_dir / state).mkdir(parents=True, exist_ok=True)
        
        # Load or create manifest
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """Load manifest or create empty one."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {"samples": []}
    
    def _save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
    
    def capture_screenshot(self) -> Optional[np.ndarray]:
        """Capture current screen."""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                # Convert to OpenCV format (BGR)
                frame = np.array(screenshot)
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def collect_sample(self, state: str, screenshot: Optional[np.ndarray] = None) -> bool:
        """
        Collect a sample for given state.
        User positions game, presses Enter when ready.
        
        Args:
            state: One of 'combat', 'narrative', 'leveling'
            screenshot: Optional pre-captured screenshot. If None, will capture on signal.
        
        Returns:
            True if successful, False otherwise.
        """
        if state not in self.states:
            print(f"Invalid state: {state}. Must be one of {self.states}")
            return False
        
        # If screenshot not provided, wait for user signal then capture
        if screenshot is None:
            print(f"Position game to show {state.upper()} state...")
            input("Press ENTER when ready to capture...")
            
            # Beep to indicate capture
            try:
                winsound.Beep(1000, 200)  # 1000Hz, 200ms
            except:
                print("*beep*")  # Fallback if winsound unavailable
            
            screenshot = self.capture_screenshot()
            if screenshot is None:
                print("Failed to capture screenshot")
                return False
        
        # Generate filename
        state_dir = self.base_dir / state
        sample_count = len(list(state_dir.glob("*.png"))) + 1
        filename = f"{state}_{sample_count:03d}.png"
        filepath = state_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), screenshot)
        
        # Update manifest
        self.manifest["samples"].append({
            "filename": str(filepath),
            "state": state,
            "timestamp": datetime.now().isoformat(),
            "size": screenshot.shape
        })
        self._save_manifest()
        
        print(f"✓ Saved {filename} to {state}/ ({sample_count} total)")
        return True
    
    def interactive_collection(self):
        """Interactive mode for collecting samples."""
        print("\n=== KOTOR UI Dataset Collector ===")
        print("States: c=combat, n=narrative, l=leveling, q=quit\n")
        
        state_map = {
            "c": "combat",
            "n": "narrative",
            "l": "leveling"
        }
        
        while True:
            choice = input("Select state (c/n/l) or q to quit: ").strip().lower()
            
            if choice == "q":
                print("Exiting...")
                break
            
            if choice not in state_map:
                print("Invalid choice")
                continue
            
            state = state_map[choice]
            self.collect_sample(state)
            print()


def main():
    parser = argparse.ArgumentParser(description="Collect KOTOR UI samples")
    parser.add_argument("--mode", choices=["collect", "batch"], default="collect",
                        help="Collection mode")
    parser.add_argument("--state", choices=["combat", "narrative", "leveling"],
                        help="State to collect (batch mode)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of samples to collect (batch mode)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between captures (batch mode)")
    
    args = parser.parse_args()
    collector = DatasetCollector()
    
    if args.mode == "collect":
        collector.interactive_collection()
    elif args.mode == "batch" and args.state:
        import time
        print(f"Collecting {args.count} samples of '{args.state}' every {args.interval}s")
        for i in range(args.count):
            remaining = args.count - i
            print(f"\nCapture {i+1}/{args.count} in {args.interval}s... ({remaining} remaining)")
            time.sleep(args.interval - 1)
            
            # Beep 1 second before capture
            try:
                winsound.Beep(800, 100)
            except:
                pass
            
            time.sleep(1)
            collector.collect_sample(args.state)
    
    print(f"\nDataset stats:")
    stats = {"combat": 0, "narrative": 0, "leveling": 0}
    for sample in collector.manifest["samples"]:
        stats[sample["state"]] += 1
    for state, count in stats.items():
        print(f"  {state}: {count}")


if __name__ == "__main__":
    main()
