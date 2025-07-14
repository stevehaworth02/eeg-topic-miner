#!/usr/bin/env python
import os, json, argparse, time, urllib.error
from typing import Dict, Optional
import orjson, ray
from dotenv import load_dotenv
from Bio import Entrez
from ray._raylet import ObjectRef 

# ─── 1.  load & export creds (driver) ─────────────────────────
load_dotenv()
for k in ("ENTREZ_EMAIL", "NCBI_API_KEY"):
    if not os.getenv(k):
        raise RuntimeError(f"{k} missing in .env")
    os.environ[k] = os.getenv(k)            # make Ray inherit

# ─── 2.  worker task (no progress bar) ───────────────────────
@ray.remote(max_retries=3)
def fetch_record(pmid: str) -> Optional[Dict]:
    import urllib.error                 
    from Bio import Entrez

    Entrez.email   = os.environ["ENTREZ_EMAIL"]
    Entrez.api_key = os.environ["NCBI_API_KEY"]

    try:
        h   = Entrez.efetch(db="pubmed", id=pmid,
                            rettype="abstract", retmode="xml")
        obj = Entrez.read(h)

        # ── robust DTD handling ─────────────────────────────
        if isinstance(obj, list) and obj:
            art = obj[0]["MedlineCitation"]["Article"]
        elif isinstance(obj, dict) and "PubmedArticle" in obj:
            art = obj["PubmedArticle"][0]["MedlineCitation"]["Article"]
        else:                         # empty or unknown shape
            return None

        abst = art.get("Abstract", {})
        return {
            "pmid": pmid,
            "title": art["ArticleTitle"],
            "abstract": " ".join(abst.get("AbstractText", [])),
            "pub_date": art["Journal"]["JournalIssue"]["PubDate"],
        }

    except urllib.error.HTTPError as e:
        if e.code in (404, 429, 500, 502, 503):
            return None
        raise
    except Exception:
        return None
    
# ─── 3.  CLI glue ────────────────────────────────────────────
def main(pmid_json: str, out_jsonl: str, workers: int = 8):
    pmids = [p for L in json.load(open(pmid_json)).values() for p in L]
    print(f"Fetching {len(pmids):,} PMIDs with {workers} workers …")

    ray.init(num_cpus=workers, include_dashboard=False)
    todo: list[ObjectRef] = [fetch_record.remote(p) for p in pmids]

    ok = skipped = 0
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "wb") as fp:
        while todo:
            done, todo = ray.wait(todo, num_returns=1, timeout=None)
            rec = ray.get(done[0])
            if rec:
                fp.write(orjson.dumps(rec) + b"\n")
                ok += 1
            else:
                skipped += 1
            if (ok + skipped) % 500 == 0:
                print(f"  processed {ok+skipped:>5}  (saved {ok}, skipped {skipped})")

    ray.shutdown()
    print(f"Done. Saved {ok} abstracts, skipped {skipped}. → {out_jsonl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    main(args.pmid_json, args.out_jsonl, args.workers)
