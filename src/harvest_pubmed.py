#!/usr/bin/env python
"""
Fetch PubMed PMIDs for each search term in config/queries.txt.
Usage:
    python src/harvest_pubmed.py \
        --query_file config/queries.txt \
        --out data/pmids.json
"""
import argparse, json, os, time
from typing import List, Dict
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()                                        # pull creds from .env
Entrez.email   = os.getenv("ENTREZ_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")

# ---- helpers ---------------------------------------------------------------

def search_pmids(term: str,
                 retmax: int = 100_000,
                 mindate: str = "2019/07/14",
                 maxdate: str = "2025/07/14") -> List[str]:
    """Return a list of PMIDs for `term` within the given date range."""
    handle = Entrez.esearch(db="pubmed",
                            term=term,
                            rettype="uilist",
                            mindate=mindate,
                            maxdate=maxdate,
                            retmax=retmax)
    record = Entrez.read(handle)
    return record["IdList"]

def harvest(query_file: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    with open(query_file) as f:
        for line in f:
            term = line.strip()
            if not term:                      # skip blank lines
                continue
            print(f"🔍  Searching: {term}")
            pmids = search_pmids(term)
            out[term] = pmids
            print(f"    → {len(pmids):,} PMIDs")
            # be polite to NCBI
            time.sleep(0.34)                  # ~3 req/s
    return out

# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = harvest(args.query_file)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(results, fp, indent=2)

    total = sum(len(v) for v in results.values())
    print(f"✅  Saved {total:,} PMIDs to {args.out}")
