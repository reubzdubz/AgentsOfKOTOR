# Vision System - KOTOR UI State Detection

## Quick Start

### 1. Collect Samples
```bash
cd vision_system
python dataset_collector.py --mode collect
```
Interactive mode—follow prompts to position KOTOR and capture each state. Creates samples in `datasets/kotor_ui_samples/`.

### 2. Collect Batch (automated)
```bash
python dataset_collector.py --mode batch --state combat --count 10 --interval 5
```
Captures 10 combat screenshots every 5 seconds.

### 3. Benchmark Classifiers
```bash
python classifier_benchmark.py --dataset datasets/kotor_ui_samples
```
Trains and evaluates all naive classifiers on your dataset. Outputs accuracy, precision, recall per state.

## Classifiers Included

- **HistogramClassifier**: Color distribution matching (fast, baseline)
- **EdgeClassifier**: Edge density-based (robust to UI layout changes)
- **EnsembleClassifier**: Voting ensemble of above

## Workflow

1. **Collect 20-30 samples per state** (combat, narrative, leveling)
2. **Run benchmark** to see baseline accuracy
3. **Iterate**: Identify misclassifications, collect more edge-case samples
4. **Choose best classifier** or threshold ensemble for production

## Dependencies

- opencv-python
- numpy
- mss (for screenshot capture)

Add to main `pyproject.toml` when ready to integrate.
