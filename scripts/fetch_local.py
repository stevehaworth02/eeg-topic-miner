# scripts/fetch_local.py
"""
Launch src.fetch_abstracts in Ray **local‑mode** so that Windows
doesn’t try to spin up a raylet or dashboard.  
Forwards all CLI args unchanged.
"""
import os
import sys
import runpy
import ray
import importlib.util

os.environ["RAY_DISABLE_DASHBOARD"] = "1"
ray.init(local_mode=True, include_dashboard=False)      # ← magic line

# ensure “src” can be imported when user runs from repo root
sys.path.insert(
    0,
    str(importlib.util.find_spec("src").submodule_search_locations[0].parent)
)

# re‑launch the real script
runpy.run_module("src.fetch_abstracts", run_name="__main__")
