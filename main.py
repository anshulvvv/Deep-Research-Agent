import asyncio
import json
import logging
from scrapy.utils.log import configure_logging

# Disable all of Scrapy’s logs right away
configure_logging({
    'LOG_ENABLED': False,
    'LOG_LEVEL': 'WARNING',
})

# Also silence related libraries
logging.getLogger("scrapy").setLevel(logging.WARNING)
logging.getLogger("twisted").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# … your logging silencing …

from interactive import interactive_plan_loop
from research    import deep_runner
from report      import generate_reports

def main():
    # synchronously drive the async pipeline under the hood
    loop = asyncio.get_event_loop()
    result_tree = loop.run_until_complete(deep_runner())
    if not result_tree:
        return

    with open("deep_research_output.json", "w", encoding="utf-8") as f:
        json.dump(result_tree, f, indent=2, sort_keys=True)
    print("Wrote deep_research_output.json")

    print("\n=== Generating Reports ===")
    generate_reports(result_tree)

if __name__ == "__main__":
    main()
