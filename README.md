# 🧠 EEG-Topic-Miner 🧠
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/container-docker-blue)](https://hub.docker.com/)
[![Made With ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)]()

*Powerful AI-driven knowledge base for neurology, deep learning, and sleep research.  
Built with reproducible, modular MLOps for easy deployment and scaling*
---
*Coming Soon: CI/CD pipeline, user can specific number of **K** abstracts, return links, titles & PMIDs rather than just PMID*
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
- [7. Video Demo](#7-Demo)

---

## 1. Background

Throughout my research endeavors, I've always sought out quality research papers in neuroscience, EEG, and sleep science. But with over 50,000 NIH publications, finding the right ones has never been harder. My labmates and I often spend hours digging through irrelevant results. As mainstream search engines became increasingly monetized, and therefore less precise, the problem only grew.
To improve this, PubMed introduced a powerful query syntax that allows researchers to filter papers more precisely. But it came with a cost: learning an unintuitive syntax that doesn't support natural search. Students and researchers now have to memorize operators just to find papers they care about.
What if finding the right paper could be as simple as asking a question?
This project builds a semantic search engine on top of the existing PubMed infrastructure. I create a structured set of domain-specific queries using PubMed’s syntax, then fine-tune a SciBERT model to recognize abstracts that match those themes. This enables a natural language search that is powered by AI while preserving the filtering precision of the underlying PubMed system. No query syntax. No wasted clicks. Just fast, relevant papers tailored to your lab's research needs.

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
### 5.1 Clone the codebase

```bash
git clone https://github.com/stevehaworth02/eeg-topic-miner.git
cd eeg-topic-miner
```

---

### 5.2 Download pretrained model (Required)

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
python3.11 -m venv .venv
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
## 6. Quick Start

Once you've completed [Installation](#5-installation), you're ready to run the full semantic search pipeline and explore EEG-related PubMed abstracts using the pretrained SciBERT model.

---

### 🐳 Docker 

This is the most reliable way to run the pipeline. Make sure you have:

- The `models/scibert_best/` folder downloaded from Hugging Face
- A valid `.env` file with your NCBI credentials

Then simply run:

```bash
docker run --rm \
  --env-file .env \
  -v "$(pwd)":/workspace \
  eeg-topic-miner
```

This will:
- Load your `.env` for PubMed access
- Use the pretrained SciBERT model (no retraining)
- Harvest → Fetch → Tokenize → Index abstracts
- Launch the interactive query tool

> 🔒 Note: This image **does not support retraining**. It will ignore edits to `config/queries.txt` unless you retrain manually

---

### 💻 macOS / Linux (Native)

If you're not using Docker, you can run the native pipeline using:

```bash
bash run_pipeline.sh
```

This will:
- Create a virtual environment (if not present)
- Install dependencies
- Load your `.env` credentials
- Harvest → Fetch → Tokenize → Index abstracts
- Launch the query tool: `src/query_faiss_index.py`

---

### 🪟 Windows (PowerShell)

For Windows users running natively, use PowerShell:

```powershell
.\run_pipeline.ps1
```

This does the same as the macOS/Linux pipeline:
- Creates and activates a virtual environment
- Installs dependencies
- Loads your `.env`
- Runs Bricks 1–4 (harvest, fetch, tokenize, index)
- Starts the interactive search tool

---

At the end of any of these options, you’ll see:

```
DONE – Ready for querying with src/query_faiss_index.py
Launching interactive search...
```

You can now enter free-text queries like:

```
> deep learning seizure detection
> sleep stage classification EEG
> brain-computer interface LSTM
```

And receive the most semantically similar neuroscience papers from the index


## 7. Demo (Coming Soon)

Click below to watch a short demo of the EEG Topic Miner in action:

[![Watch the demo](https://img.youtube.com/vi/your_video_id/hqdefault.jpg)](https://www.youtube.com/watch?v=your_video_id)

*“Progress isn’t born from comfort — it’s coaxed from chaos and debugged into elegance.”*


