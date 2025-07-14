# sanity.py
from Bio import Entrez
import json, os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

Entrez.email, Entrez.api_key = os.getenv("ENTREZ_EMAIL"), os.getenv("NCBI_API_KEY")

pmid = json.load(open("data/pmids.json"))["eeg[tiab] AND seizure detection"][0]
print("Testing PMID:", pmid)

handle = Entrez.efetch(db="pubmed", id=pmid,
                       rettype="abstract", retmode="xml")
obj = Entrez.read(handle)

# ---- handle both shapes ---------------------------------------------------
if isinstance(obj, list) and obj:
    art = obj[0]["MedlineCitation"]["Article"]
elif isinstance(obj, dict) and "PubmedArticle" in obj:
    art = obj["PubmedArticle"][0]["MedlineCitation"]["Article"]
else:
    raise RuntimeError("Unknown XML shape")

print("Title:", art["ArticleTitle"][:80], "…")
