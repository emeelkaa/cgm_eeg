# CGM-EEG: Cross-Gated Mamba for Spatio-Temporal EEG Representation Learning

This repository contains the official implementation of CGM-EEG, a fully Mamba-based dual-branch architecture for efficient and accurate EEG decoding. CGM-EEG models temporal and spatial dependencies in parallel using bidirectional Mamba encoders and a lightweight Cross-Gate Module (CGM) that enables dynamic feature interaction across branches.

✨ Key Features

Dual-branch Temporal–Spatial Mamba Encoder for joint EEG sequence modeling

Cross-Gate Module for efficient bidirectional feature fusion

Supports binary and multi-class EEG classification (CHB-MIT, TUEV, TUSZ)

Implements subject-dependent and subject-independent evaluation protocols

Fully reproducible PyTorch training and evaluation pipeline

📊 Results
CGM-EEG achieves up to 3.5% higher balanced accuracy and 7.1% lower inference latency than recent Transformer-based EEG models, while maintaining strong generalization across datasets.

## Files
- `train.py`: Main training pipeline.
- `dataset.py`: Dataset class and preprocessing functions.
- `models/`: Contains model definitions.

## Usage
```bash
python train.py
# HSST-EEG

