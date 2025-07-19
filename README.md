# 🧠 EEG-Topic-Miner 🧠

*Powerful AI-Driven knowledge base for neurology, deep learning, and sleep research... CI/CD coming soon*

---

# Table of Contents

- [0. Disclaimer](#0-disclaimer)
  - [Bugs & Contact](#bugs--contact)
  - [Training Your Own Model](#training-your-own-model)
- [1. Background](#1-background)
- [2. Features](#2-features)
- [3. Installation](#3-installation)
- [4. Quick Start](#4-quick-start)
- [5. FAQ](#5-faq)
- [6. Acknowledgements](#6-acknowledgements)

---

## 0. Disclaimer

### Bugs & Contact

If you find a bug, issue, or something isn’t working as expected, **please don’t panic!**  
Reach out at [0218steven@gmail.com](mailto:0218steven@gmail.com) and I’ll help as soon as possible.

### Training Your Own Model

Want to train your own model? Here’s what to do:

- You’ll need **significant compute power** (expect very long runtimes on CPU, recommend using a CUDA-compatible GPU).
- The training pipeline is set up in `run_pipeline.ps1` (Windows/PowerShell) and `run_pipeline.sh` (Linux/Mac/Docker).
- To retrain:  
  - Make sure your data is prepared and in the correct format.
  - Run the training step manually:  
    - On Windows:  
      ```
      python scripts/train_serial.py --data_dir data/tokenised --out_dir models/scibert_best
      ```
    - On Linux/Mac:  
      ```
      python src/train.py --data_dir data/tokenised --out_dir models/scibert_best
      ```
- **Warning:** Training on CPU may take hours; CUDA-compatible GPUs are strongly recommended for reasonable speed.

---

## 1. Background

*To be completed!*

---

