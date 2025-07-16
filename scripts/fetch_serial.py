#!/usr/bin/env python
"""
Drop‑in replacement for fetch_abstracts that uses plain
requests (no Ray) so it works on Windows without raylet.
CLI flags are identical to the original script.
"""
import argparse, json, os, urllib.error
from typing import Dict, Optional
import orjson
from dotenv import load_dotenv
from Bio import Entrez
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── creds ──────────────────────────────────────────────────
load_dotenv()
Entrez.email   = os.getenv("ENTREZ_EMAIL")
Entrez.api_key = os.getenv("NCBI_API_KEY")
if not Entrez.email or not Entrez.api_key:
    raise RuntimeError("ENTREZ_EMAIL / NCBI_API_KEY missing in .env")

# ── one request ────────────────────────────────────────────
def fetch(pmid: str) -> Optional[Dict]:
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid,
                               rettype="abstract", retmode="xml")
        obj = Entrez.read(handle)

        # try the two shapes we know about, but only if they have data
        if isinstance(obj, list) and obj:
            art = obj[0]["MedlineCitation"]["Article"]
        elif isinstance(obj, dict) and obj.get("PubmedArticle"):
            articles = obj["PubmedArticle"]
            if not articles:
                return None
            art = articles[0]["MedlineCitation"]["Article"]
        else:
            return None

        abst = art.get("Abstract", {})
        return {
            "pmid":     pmid,
            "title":    art["ArticleTitle"],
            "abstract": " ".join(abst.get("AbstractText", [])),
            "pub_date": art["Journal"]["JournalIssue"]["PubDate"],
        }

    except urllib.error.HTTPError as e:
        # transient errors we skip
        if e.code in (404, 429, 500, 502, 503):
            return None
        raise
    except Exception:
        # any other parsing oddity, just skip
        return None

# ── CLI ────────────────────────────────────────────────────
def main(pmid_json: str, out_jsonl: str, workers: int = 8):
    pmids = [p for L in json.load(open(pmid_json)).values() for p in L]
    print(f"Fetching {len(pmids):,} PMIDs with {workers} threads …")

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    ok = skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as pool, \
         open(out_jsonl, "wb") as fp:

        futs = {pool.submit(fetch, p): p for p in pmids}
        for fut in as_completed(futs):
            rec = fut.result()
            if rec:
                fp.write(orjson.dumps(rec) + b"\n")
                fp.flush()           # <— force write immediately
                ok += 1
            else:
                skipped += 1

            # print progress every 50
            total = ok + skipped
            if total % 50 == 0:
                print(f"  processed {total:>5}  (saved {ok}, skipped {skipped})", flush=True)

    print(f"Done. Saved {ok} abstracts, skipped {skipped}. → {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--workers",  type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()
    main(args.pmid_json, args.out_jsonl, args.workers)
