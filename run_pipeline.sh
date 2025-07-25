#!/usr/bin/env bash
# run_pipeline.sh -- Cross‑platform driver for Bricks 1–4 (now only harvests, fetches, tokenizes, indexes)
# For Mac/Linux/Unix (non-Windows). Requires Python 3.11.

set -euo pipefail

# ─── Choose fetcher & default parallelism ───────────────────────────
fetchScript="src/fetch_abstracts.py"
tokScript="src/preprocess_tokenize.py"
maxWorkers=8

# ---------- Python & venv ------------------------------------------------
pyCmd="python3.11"
venv=".venv"

if [[ ! -d "$venv" ]]; then
  $pyCmd -m venv $venv
fi

source "$venv/bin/activate"

pip install -U pip
pip install -r requirements.txt "numpy<2" numexpr

# ---------- Load .env ------------------------------------------------------
if [[ ! -f .env ]]; then
  echo ".env missing - add ENTREZ_EMAIL and NCBI_API_KEY"
  exit 1
fi

# Export variables from .env
set -a
source .env
set +a

if [[ -z "${ENTREZ_EMAIL:-}" || -z "${NCBI_API_KEY:-}" ]]; then
  echo "ENTREZ_EMAIL or NCBI_API_KEY not set in .env"
  exit 1
fi

# ---------- Brick 1: harvest PMIDs -----------------------------------------
if [[ ! -f "data/pmids.json" ]]; then
  echo ">> Brick 1: harvest_pubmed.py"
  python src/harvest_pubmed.py --query_file config/queries.txt --out data/pmids.json
else
  echo "Brick 1 already done"
fi

# ---------- Brick 2: fetch abstracts ---------------------------------------
if [[ ! -f "data/raw.jsonl" ]]; then
  echo ">> Brick 2: fetch abstracts via $fetchScript"
  python $fetchScript \
    --pmid_json data/pmids.json \
    --out_jsonl data/raw.jsonl \
    --workers   $maxWorkers
else
  echo "Brick 2 already done"
fi

# ---------- Brick 3: tokenize & preprocess ---------------------------------
if [[ ! -d "data/tokenised" ]]; then
  echo ">> Brick 3: tokenize via $tokScript"
  python $tokScript \
    --raw_jsonl data/raw.jsonl \
    --out_dir   data/tokenised \
    --workers   $maxWorkers
else
  echo "Brick 3 already done"
fi

# ---------- Brick 3.5: Fine-tune SciBERT (Optional, commented out) ---------
# Uncomment this to retrain the model on your own data (requires a CUDA GPU!)
# echo ">> Brick 3.5: Training model (this may take hours on CPU!)"
# python src/train.py \
#   --data_dir data/tokenised \
#   --out_dir  models/scibert_best

# ---------- Brick 4: build FAISS index (no retraining) ---------------------
if [[ ! -f "models/scibert_best/faiss.index" ]]; then
  echo ">> Brick 4: build_faiss_index.py (indexing with pretrained model)"
  python src/build_faiss_index.py
else
  echo "Brick 4 already done"
fi

echo ""
echo "DONE - Ready for querying with src/query_faiss_index.py"
echo ""
echo "NOTE: Model retraining is disabled by default. To retrain, run scripts/train_serial.py manually."
echo ""
echo "Launching interactive search (you can exit anytime)..."
python src/query_faiss_index.py
