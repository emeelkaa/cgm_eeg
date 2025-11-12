# 🧠 CGM-EEG: Cross-Gated Mamba for Spatio-Temporal EEG Representation Learning

Official PyTorch implementation of **CGM-EEG**, a fully Mamba-based dual-branch architecture for efficient and accurate EEG decoding.  

---

## 📘 Overview

![Framework Overview](pipeline.png)

**CGM-EEG** introduces a *dual-branch design* combining temporal and spatial Mamba encoders, where each branch captures complementary dependencies.  
The **Cross-Gate Module (CGM)** enables bidirectional interaction between the two branches, refining the learned representations.
This approach achieves high decoding accuracy and reduced inference latency across multiple EEG benchmarks.

Extensive experiments on three public clinical EEG benchmarks demonstrate that CGM-EEG achieves 
up to 6.6% higher balanced accuracy and 7.1% lower inference latency than recent transformer-based models.

---

## 📂 Repository Structure
```
CGM-EEG/
├── dataset.py              # Dataset loading and preprocessing utilities
├── train.py                # Main training script
├── requirements.txt        # Python dependencies
├── models/                 # Model architectures
│   ├── __init__.py
│   ├── cgm_eeg.py         # Main CGM-EEG model
│   ├── eeg_conformer.py   # Song, Yonghao, et al. (2022)
│   └── biot.py            # Yang, Chaoqi, et al. (2023)
│   └── sparcnet.py        # Jing, Jin, et al. (2023)
│   └── tsception.py       # Ding, Yi, et al. (2022)
├── preprocessing/          # Data preprocessing scripts
│   ├── will commit later
├── pipeline.png            # Architecture diagram
├── README.md               # This file
```
