#!/usr/bin/env bash
set -euo pipefail
#
#   run_pipeline.sh  – One‑shot driver for Bricks 1‑4
#   • Creates .venv (3.11) if missing
#   • Installs/updates requirements (pins numpy<2)
#   • Runs harvest_pubmed.py  ➜ data/pmids.json         (Brick 1)
#   • Runs fetch_abstracts.py ➜ data/raw.jsonl          (Brick 2)
#   • Runs preprocess_tokenize.py ➜ data/tokenised/     (Brick 3)
#   • Runs train.py             ➜ models/scibert_best/ (Brick 4)
#

PY=python3.11               # ensure 3.11 is on PATH
VENV=.venv

echo "▶ checking virtual‑env …"
if [[ ! -d "$VENV" ]]; then
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "▶ installing requirements …"
pip install -U pip
pip install -r requirements.txt "numpy<2" numexpr

# --- sanity‑check ENV creds ----------------------------------------------------
[[ -f .env ]] && source .env
: "${ENTREZ_EMAIL?Need ENTREZ_EMAIL in .env}"
: "${NCBI_API_KEY?Need NCBI_API_KEY in .env}"

# -----------------------------------------------------------------------------#
# Brick 1 – Harvest PMIDs
# -----------------------------------------------------------------------------#
if [[ ! -f data/pmids.json ]]; then
  echo "▶ Brick 1: harvesting PMIDs …"
  python src/harvest_pubmed.py \
         --query_file config/queries.txt \
         --out data/pmids.json
else
  echo "✓ Brick 1: data/pmids.json already exists – skipping"
fi

# -----------------------------------------------------------------------------#
# Brick 2 – Fetch abstracts
# -----------------------------------------------------------------------------#
if [[ ! -f data/raw.jsonl ]]; then
  echo "▶ Brick 2: fetching abstracts …"
  python src/fetch_abstracts.py \
         --pmid_json data/pmids.json \
         --out_jsonl data/raw.jsonl \
         --workers 8
else
  echo "✓ Brick 2: data/raw.jsonl already exists – skipping"
fi

# -----------------------------------------------------------------------------#
# Brick 3 – Tokenise + label flags
# -----------------------------------------------------------------------------#
if [[ ! -d data/tokenised ]]; then
  echo "▶ Brick 3: tokenising dataset …"
  python src/preprocess_tokenize.py \
         --raw_jsonl data/raw.jsonl \
         --out_dir  data/tokenised \
         --workers 8
else
  echo "✓ Brick 3: data/tokenised/ already exists – skipping"
fi

# -----------------------------------------------------------------------------#
# Brick 4 – Fine‑tune SciBERT
# -----------------------------------------------------------------------------#
echo "▶ Brick 4: launching Ray Tune sweep …"
python src/train.py \
       --data_dir data/tokenised \
       --out_dir  models/scibert_best

echo "🎉  Pipeline finished – model in models/scibert_best/"
