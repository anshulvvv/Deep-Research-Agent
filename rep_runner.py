import json
import os
import logging

# silence any unwanted logs
from scrapy.utils.log import configure_logging
configure_logging({'LOG_ENABLED': False, 'LOG_LEVEL': 'WARNING'})
logging.getLogger("scrapy").setLevel(logging.WARNING)
logging.getLogger("twisted").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# make sure cache.py and report.py are on your PYTHONPATH
from cache import load_cache
from report import generate_reports

def main():
    # load the cache (must match what load_cache() expects)
    CACHE = load_cache()
    print(f"Loaded cache with {len(CACHE)} entries")

    # load the saved research tree
    tree_path = "deep_research_output.json"
    if not os.path.exists(tree_path):
        raise RuntimeError(f"{tree_path!r} not found. Run deep_runner first.")
    with open(tree_path, "r", encoding="utf-8") as f:
        result_tree = json.load(f)
    print(f"Loaded research tree with {len(result_tree['tasks'])} tasks")

    # now generate your combined JSON and Markdown
    generate_reports(result_tree)

if __name__ == "__main__":
    main()