# scripts/fetch_local.py
"""
Run src.fetch_abstracts in Ray *local‑mode* so Windows doesn’t spin up
a Raylet or dashboard.  Forwards all CLI args unchanged.
"""
import runpy, ray, sys, os, importlib.util

# disable the dashboard even if someone forgets the env‑var
os.environ["RAY_DISABLE_DASHBOARD"] = "1"
ray.init(local_mode=True, include_dashboard=False)

# make sure relative import of src.fetch_abstracts works
# if the user runs from repo root
sys.path.insert(0, str(importlib.util.find_spec("src").submodule_search_locations[0].parent))

# re‑launch the real script
runpy.run_module("src.fetch_abstracts", run_name="__main__")
