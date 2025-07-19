# 🧠 EEG-Topic-Miner 🧠

*Powerful AI-driven knowledge base for neurology, deep learning, and sleep research.  
Built with reproducible, modular MLOps for easy deployment and scaling—CI/CD coming soon!*

---

# Table of Contents

- [1. Background](#1-background)
- [2. Technology](#2-technology)
- [3. Disclaimer](#3-disclaimer)
  - [Bugs & Contact](#bugs--contact)
  - [Training Your Own Model](#training-your-own-model)
- [4. Features](#4-features)
- [5. Installation](#5-installation)
- [6. Quick Start](#6-quick-start)
- [7. FAQ](#7-faq)
- [8. Acknowledgements](#8-acknowledgements)

---

## 1. Background

Throughout my research endeavors, I've always sought out quality research papers within my field. With over 50,000 NIH publications, finding what I need has never been harder. Me and my labmates spend significant time digging for relevant papers. As mainstream search engines became increasingly monetized and therefore less precise, the problem grew. What if finding the right paper could be as simple as asking a question?  
This project aims to provide just that: a fast, semantic search engine for the latest neuroscience, EEG, and sleep research, powered by deep learning and AI.

---

## 2. Technology

- **Automated, Modular Pipeline:**  
  Each step (data harvesting, preprocessing, training, indexing, querying) is script-driven and reproducible, making updates and debugging simple.

- **Cross-Platform & Containerization:**  
  Runs on Windows, Mac, or Linux—and **Docker** support means users don’t have to fight with dependencies or CUDA compatibility.

- **Pretrained Models and Artifacts:**  
  All large artifacts (trained models, FAISS indexes, tokenized datasets) are hosted on Hugging Face, keeping the repo clean and speeding up setup.

- **CI/CD-Ready:**  
  The design allows future integration with continuous integration and deployment pipelines—automating retraining, evaluation, and deployment as new data comes in.

- **User-Facing AI Search:**  
  Our semantic search uses **SciBERT** to encode user queries and abstracts, with FAISS for blazing-fast similarity search across thousands of publications.

---

## 3. Disclaimer

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
