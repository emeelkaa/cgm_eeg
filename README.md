# 🧠 CGM-EEG: Cross-Gated Mamba for Spatio-Temporal EEG Representation Learning

Official PyTorch implementation of **CGM-EEG**, a fully Mamba-based dual-branch architecture for efficient and accurate EEG decoding.  
CGM-EEG models temporal and spatial dependencies in parallel using bidirectional Mamba encoders and a lightweight **Cross-Gate Module (CGM)** that enables dynamic information exchange between branches.

---

## 📘 Overview

![Framework Overview](fig/pipeline.png)

**CGM-EEG** introduces a *dual-branch design* combining temporal and spatial Mamba encoders, where each branch captures complementary dependencies.  
The **Cross-Gate Module (CGM)** integrates contextual cues across branches, improving representational power while maintaining linear-time efficiency.  
This approach achieves high decoding accuracy and reduced inference latency across multiple EEG benchmarks.

---

## 📂 Repository Structure
CGM-EEG/
│
├── dataset.py # EEG dataset loaders and preprocessing pipelines
├── models/ # Implementations of CGM-EEG and baseline models (BIOT, TSception, EEG-Conformer, SPaRCNet)
├── train.py # Main training scripts for all experimental settings
├── requirements.txt # Python dependencies
└── README.md # Project documentation
