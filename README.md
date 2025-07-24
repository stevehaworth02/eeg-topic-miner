# 🧠 EEG-Topic-Miner 🧠
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/container-docker-blue)](https://hub.docker.com/)
[![Made With ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)]()

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

---

## 1. Background

Throughout my research endeavors, I've always sought out quality research papers within my field. With over 50,000 NIH publications, finding what I need has never been harder. Me and my labmates spend significant time digging for relevant papers. As mainstream search engines became increasingly monetized and therefore less precise, the problem grew. What if finding the right paper could be as simple as asking a question?  

This project aims to provide just that: a fast, semantic search engine for the latest neuroscience, EEG, and sleep research, powered by deep learning and AI. Furthermore, for Systematic Reviews, a user can easily input their PubMed queries into queries.txt to retrain the model and recieve more personalized results while maintaining the filtering 

---

## 2. Technology

- **Automated, Modular Pipeline:**  
  Each step (data harvesting, preprocessing, training, indexing, querying) is scripted and reproducible, making updates and debugging simple.

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

Model retraining is **disabled by default** for user-friendliness, reliability, and fast setup.

**If you want to train your own model:**
- **Uncomment** the training command in your platform’s pipeline file:
  - In `run_pipeline.ps1` (Windows/PowerShell)
  - In `run_pipeline.sh` (Linux/Mac/Docker)
- Or, you may run the training script manually:
  - **Windows:**
    ```powershell
    python scripts/train_serial.py --data_dir data/tokenised --out_dir models/scibert_best
    ```
  - **Linux/Mac/Docker:**
    ```bash
    python src/train.py --data_dir data/tokenised --out_dir models/scibert_best
    ```
- You’ll need **significant compute power** (expect long runtimes on CPU; CUDA-compatible GPUs are strongly recommended).
- **Note:** By default, the pipeline will use the latest pretrained models and indexes hosted on Hugging Face.

**On Mac/Linux:**

- Run with **Docker** for maximum reliability (recommended for most users; handles all dependencies for you), or
- Run with `run_pipeline.sh` if you want a **native setup** (runs directly on your system’s Python, best for users comfortable with installing dependencies).

- **Warning:** Training on CPU may take hours; CUDA-compatible GPUs are strongly recommended for reasonable training speed (Queries take milliseconds though!).

---

## 4. Features

- **Domain-Fine-Tuned Model:**  
  My SciBERT model was fine-tuned on EEG, sleep, and neurology abstracts, so you get results tailored to your domain rather not just generic science.

- **Semantic Matching (Fine-Tuned!):**  
  Uses a fine-tuned SciBERT model trained specifically for EEG, neurology, and sleep literature—plus FAISS, to surface papers similar in meaning, not just text overlap. Results are actually relevant to your research, rather than just linguistically similar.

- **Blazing-Fast Search:**  
  Query thousands of abstracts in milliseconds with FAISS, even on a laptop.

- **Zero Setup Headaches:**  
  Get started instantly: use Docker or PowerShell for no CUDA or messy installs.

- **Automatic Metadata & Weak Labels:**  
  Each abstract is auto-tagged by topic (seizure, sleep, BCI, or unlabeled) and flagged for deep learning content which enable smart filters and fast dataset exploration.

- **Reproducible, Modular Pipeline:**  
  Data harvesting, preprocessing, training, indexing, and querying are all broken into clear, reusable scripts—easy to update, debug, or extend.

- **Cross-Platform (Windows, Mac, Linux):**  
  Search, build, and query on any OS, just choose the right pipeline or Docker image.

- **Future-Proof & CI/CD Ready:**  
  The architecture is designed for continuous improvement: we can plug in new data, re-index, or retrain as needed (or just query if you prefer!).

- **Open Source, Fully Documented:**  
  All code, scripts, and guides are open and well-commented, so you can learn, adapt, or contribute.

---
## 5. Installation

### 5.0 Requirements

Before installation, ensure the following are installed on your system:

- [Python 3.11+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/) (for containerized setup)
- [Git](https://git-scm.com/) (for cloning the repos)

---

### 5.1 Download pretrained model (Required)

This project uses a pretrained SciBERT model hosted on Hugging Face. You must manually download it **before building the Docker image or running the native pipeline**.

```bash
# Clone the pretrained model repo into models/
git clone https://huggingface.co/stevehaworth02/eeg-topic-miner-model models/scibert_best
```

> 📁 Make sure the cloned folder is located at:  
> `./models/scibert_best/` inside your repo root.

This folder should contain:
- `best_model.pt`
- `faiss.index`
- `faiss_meta.npy`

---

### 5.2 Clone the codebase

```bash
git clone https://github.com/stevehaworth02/eeg-topic-miner.git
cd eeg-topic-miner
```

---

### 5.3 Create your credentials file
**Note:** Steps for attaining an API-Key are here: https://support.nlm.nih.gov/kbArticle/?pn=KA-05317
```bash
cp .env.example .env
# then edit .env and add:
#   ENTREZ_EMAIL=your@email
#   NCBI_API_KEY=your_key
```

---

### 5.4 Native install (macOS / Linux)

```bash
# 1) Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt "numpy<2" numexpr
```

---

### 5.5 Native install (Windows / PowerShell)

```powershell
# 1) Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt "numpy<2" numexpr
```

---

### 5.6 Docker install (Recommended)

This method runs the **fully pretrained version only** — model retraining is not included in the image.

```bash
# Build the Docker image (make sure models/scibert_best exists!)
docker build -t eeg-topic-miner .

# Run the container (requires .env in current directory)
docker run --rm \
  --env-file .env \
  -v "$(pwd)":/workspace \
  eeg-topic-miner
```

---








