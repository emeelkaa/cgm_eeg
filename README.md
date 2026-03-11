# 🧠 CGM-EEG: Cross-Gated Mamba for Spatio-Temporal EEG Representation Learning [ISBI 2026]

Official PyTorch implementation of **CGM-EEG**, a fully Mamba-based dual-branch architecture for efficient and accurate EEG decoding. **CGM-EEG** introduces a *dual-branch design* combining temporal and spatial Mamba encoders, where each branch captures complementary dependencies.  
The **Cross-Gate Module (CGM)** enables bidirectional interaction between the two branches, refining the learned representations.
Extensive experiments on three public clinical EEG benchmarks demonstrate that CGM-EEG achieves 
up to 6.6% higher balanced accuracy and 7.1% lower inference latency than recent transformer-based models.

## 📘 Overview
![Framework Overview](assets/pipeline.png)
---

## 📂 Repository Structure
```
CGM-EEG/
├── assets/                 # Miscellaneous files
│   ├── pipeline.png         # Architecture diagram
├── models/                 # Model architectures
│   ├── cgm_eeg.py            # Main CGM-EEG model
│   ├── eeg_conformer.py      # Song, Yonghao, et al. (2022)
│   ├── biot.py               # Yang, Chaoqi, et al. (2023)
│   ├── sparcnet.py           # Jing, Jin, et al. (2023)
│   ├── tsception.py          # Ding, Yi, et al. (2022)
├── preprocessing/          # Data preprocessing scripts
│   ├── chbmit/               # CHBMIT preprocessing
│   ├── tuev/                 # TUEV preprocessing
├── get_dataset.py          # Dataset loading
├── main.py                 # Main training script
├── engine.py               # Training engine
├── optim.py                # Optimizer 
├── utils.py                # Utilities (e.g., metric logging)
├── requirements.txt        # Python dependencies
├── README.md               # This file
```
### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/emeelkaa/cgm_eeg.git
   cd CGM-EEG
```

2. **Create a virtual environment (recommended)**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```
> **📌 Note:** For Mamba installation, please refer to the official repository:
> [https://github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)
## 📧 Contact

For questions, issues, or collaboration inquiries, please contact:

- **Email**: [emilkim01@pusan.ac.kr](mailto:emilkim01@pusan.ac.kr)
- **Author**: Emil Kim

---
