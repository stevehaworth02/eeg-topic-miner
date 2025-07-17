# run_pipeline.ps1  --  Cross‑platform driver for Bricks 1–4
# Requires PowerShell 5+ (Windows) or PowerShell Core (macOS/Linux)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Choose fetcher & default parallelism ───────────────────────────
if ($Env:OS -eq 'Windows_NT') {
  $fetchScript = "scripts\fetch_serial.py"
  $maxWorkers  = 1
} else {
  $fetchScript = "src/fetch_abstracts.py"
  $maxWorkers  = 8
}

# ─── Disable Ray dashboard on Windows (harmless elsewhere) ───────────
$env:RAY_DISABLE_DASHBOARD = "1"

# ---------- Python & venv ------------------------------------------------
$pyCmd = "python"         # must be Python 3.11
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

# ---------- Load .env ------------------------------------------------------
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

# ---------- Brick 1: harvest PMIDs -----------------------------------------
if (-not (Test-Path "data/pmids.json")) {
  Write-Host ">> Brick 1: harvest_pubmed.py"
  & python src/harvest_pubmed.py --query_file config/queries.txt --out data/pmids.json
} else {
  Write-Host "Brick 1 already done"
}

# ---------- Brick 2: fetch abstracts ---------------------------------------
if (-not (Test-Path "data/raw.jsonl")) {
  Write-Host ">> Brick 2: fetch abstracts via $fetchScript"
  & python $fetchScript `
      --pmid_json data/pmids.json `
      --out_jsonl data/raw.jsonl `
      --workers   $maxWorkers
} else {
  Write-Host "Brick 2 already done"
}

# Brick 3: tokenize & preprocess
if ($Env:OS -eq 'Windows_NT') {
  $tokScript = "scripts\preprocess_tokenize_local.py"
} else {
  $tokScript = "src/preprocess_tokenize.py"
}
if (-not (Test-Path "data/tokenised")) {
  Write-Host ">> Brick 3: preprocess_tokenize via $tokScript"
  & python $tokScript `
      --raw_jsonl data/raw.jsonl `
      --out_dir   data/tokenised `
      --workers   $maxWorkers
} else {
  Write-Host "Brick 3 already done"
}

# ---------- Brick 4: train w/ Ray Tune ------------------------------------
Write-Host ">> Brick 4: train.py (Ray Tune sweep)"
& python src/train.py --data_dir data/tokenised --out_dir models/scibert_best

Write-Host ""
Write-Host "DONE ‑ model saved to models/scibert_best" -ForegroundColor Green
