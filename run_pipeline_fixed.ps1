# run_pipeline.ps1 -- Cross-platform driver for Bricks 1-4 (now only harvests, fetches, tokenizes, indexes)
# Requires PowerShell 5+ (Windows) or PowerShell Core (macOS/Linux)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Choose fetcher & default parallelism
if ($Env:OS -eq 'Windows_NT') {
  $fetchScript = "scripts\fetch_serial.py"
  $tokScript   = "scripts/tokenize_local.py"
  $maxWorkers  = 1
} else {
  $fetchScript = "src/fetch_abstracts.py"
  $tokScript   = "src/preprocess_tokenize.py"
  $maxWorkers  = 8
}

$env:RAY_DISABLE_DASHBOARD = "1"

# Python & venv
$pyCmd = "python"
$venv  = ".venv"

if (-not (Test-Path $venv)) {
  & $pyCmd -m venv $venv
}

if ($Env:OS -eq 'Windows_NT') {
  & "$venv\Scripts\Activate.ps1"
} else {
  . "$venv/bin/activate"
}

pip install -U pip
pip install -r requirements.txt "numpy<2" numexpr

# Load .env
if (-not (Test-Path ".env")) {
  throw ".env missing - add ENTREZ_EMAIL and NCBI_API_KEY"
}
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#=]+?)\s*=\s*(.+)$') {
    Set-Item "Env:$($matches[1])" $matches[2]
  }
}
if (-not $Env:ENTREZ_EMAIL -or -not $Env:NCBI_API_KEY) {
  throw "ENTREZ_EMAIL or NCBI_API_KEY not set in .env"
}

# Brick 1: harvest PMIDs
if (-not (Test-Path "data/pmids.json")) {
  Write-Host ">> Brick 1: harvest_pubmed.py"
  & python src/harvest_pubmed.py --query_file config/queries.txt --out data/pmids.json
} else {
  Write-Host "Brick 1 already done"
}

# Brick 2: fetch abstracts (skipped - expects Hugging Face data clone)
if (-not (Test-Path "data/raw.jsonl")) {
  Write-Host "data/raw.jsonl not found. Please clone the data repo:"
  Write-Host "    git clone https://huggingface.co/datasets/sehaworth/eeg-topic-miner-data data" -ForegroundColor Yellow
  exit 1
} else {
  Write-Host "Brick 2 skipped - using pre-fetched data/raw.jsonl from Hugging Face"
}

# Brick 3: tokenize & preprocess
if (-not (Test-Path "data/tokenised")) {
  Write-Host ">> Brick 3: tokenize via $tokScript"
  & python $tokScript `
      --raw_jsonl data/raw.jsonl `
      --out_dir   data/tokenised `
      --workers   $maxWorkers
} else {
  Write-Host "Brick 3 already done"
}

# Brick 3.5: Fine-tune SciBERT (Optional, commented out)
# Uncomment this to retrain the model on your own data (requires a CUDA GPU!)
# Write-Host ">> Brick 3.5: Training model (this may take hours on CPU!)"
# & python scripts/train_serial.py --data_dir data/tokenised --out_dir models/scibert_best

# Brick 4: build FAISS index (no retraining)
if (-not (Test-Path "models/scibert_best/faiss.index")) {
  Write-Host ">> Brick 4: build_faiss_index.py (indexing with pretrained model)"
  & python src/build_faiss_index.py
} else {
  Write-Host "Brick 4 already done"
}

Write-Host ""
Write-Host "DONE - Ready for querying with src/query_faiss_index.py" -ForegroundColor Green

Write-Host ""
Write-Host "NOTE: Model retraining is disabled by default. To retrain, run scripts/train_serial.py manually."

Write-Host ""
Write-Host "Launching interactive query, you can leave any time..."
& python src/query_faiss_index.py