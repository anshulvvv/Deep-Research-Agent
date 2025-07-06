import re
from urllib.parse import urlparse
import asyncio
import json
from typing import Optional, Set, List, Dict, Any
import arxiv
from bs4 import BeautifulSoup
import nest_asyncio
from io import BytesIO
from requests.exceptions import HTTPError
from requests import Session
from PyPDF2 import PdfReader
from pydantic import BaseModel,Field
from langchain.callbacks.base import BaseCallbackHandler
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from config import settings
from tools import search_web, fetch_url, fetch_pdf, fetch_arxiv
from spider import fetch_papers
from scrapy.utils.log import configure_logging
from interactive import interactive_plan_loop
import logging
from typing import Literal, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import cache
import os
import uuid

def random_filename(extension: str = "") -> str:
    # uuid4 gives a random 128-bit GUID
    name = uuid.uuid4().hex
    if extension and not extension.startswith("."):
        extension = "." + extension
    return name + extension

# disable all of Scrapy’s own logs
configure_logging({
    'LOG_ENABLED': False,
    'LOG_LEVEL':   'WARNING',
})
# ─── 2) Token-streaming callback ──────────────────────────────────────────────
class PrintCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)
# Enable nested asyncio loops
nest_asyncio.apply()
# ─── 3) Instantiate LLM & structured interfaces ───────────────────────────────

llm = ChatGoogleGenerativeAI(
    model=settings.MODEL,
    google_api_key=settings.GEMINI_API_KEY,
    disable_streaming=False,
    callbacks=[PrintCallbackHandler()],
    verbose=True
)
# ─── 1) Structured-output models ──────────────────────────────────────────────
class AnswerWithJustification(BaseModel):
    answer: str
    justification: str
structured_llm = llm.with_structured_output(AnswerWithJustification)

class SubQ(BaseModel):
    answer: List[str]
    justification: str

class SubQ1(BaseModel):
    answer: str
    justification: str
    isSummary: bool = Field(
        ..., description="Whether the text contains anything to summarise or not "
    )

class SourceSelection(BaseModel):
    """Which source type to use and how many items (max 5) to fetch."""
    search_preference: Literal["web", "arxiv", "mixed"] = Field(
        ..., description="Whether to favor web, arXiv, or both"
    )
    web_limit: int = Field(
        ..., ge=0, le=5, description="Number of web pages to scrape (max 5)"
    )
    arxiv_limit: int = Field(
        ..., ge=0, le=5, description="Number of arXiv papers to fetch (max 5)"
    )
    justification: str = Field(
        ..., description="LLM's reasoning for the source selection"
    )




struct_llm     = llm.with_structured_output(SubQ)
struct2_llm     = llm.with_structured_output(SubQ1)
select_llm    = llm.with_structured_output(SourceSelection)
# ─── 4) In-memory cache & regexes ──────────────────────────────────────────────

ARXIV_ABS_RE = re.compile(r'arxiv\.org/abs/([^/?]+)')
ARXIV_ID_RE  = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
ARXIV_RE = re.compile(
    r"""^/(?:abs|pdf)/
        (?P<id>\d{4}\.\d{4,5})(?:v\d+)?   # e.g. 2101.01234 or 2101.01234v2
        (?:\.pdf)?$                        # optional “.pdf” suffix
    """,
    re.VERBOSE
)
session = Session()
LOG_FILE = "cot.log"
def log_thought(just: str):
    print(f"[DEEP RESEARCH] {just}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[DEEP RESEARCH] {just}\n")
def is_arxiv_url(url: str) -> bool:
    p = urlparse(url)
    if p.netloc not in ("arxiv.org", "www.arxiv.org", "export.arxiv.org"):
        return False
    return bool(ARXIV_RE.match(p.path))

async def fetch_and_cache(src: str) -> str:
    if src in cache.CACHE:
        return cache.CACHE[src]

    text = ""
    m = ARXIV_ABS_RE.search(src)
    if m and ARXIV_ID_RE.match(m.group(1)):
        try:
            meta = fetch_arxiv(m.group(1))
            text = meta.get('abstract', '')
        except HTTPError as e:
            print(f"[WARN] fetch_arxiv({m.group(1)}) failed: {e}")
    if not text and src.lower().endswith('.pdf'):
        text = fetch_pdf(src).get('text', '')
    if not text and src.startswith('http'):
        try:
            text = fetch_url(src).get('text', '')
        except Exception as e:
            print(f"[WARN] fetch_url({src}) failed: {e}")

    cache.CACHE[src] = text
    
    return text


def summarize_src(src: str) -> tuple[str, str]:
    
    entry = cache.CACHE.get(src, {})
    if not entry.get("text") and not entry.get("abstract"):
        #log_thought(f"No content for {src}")
        return src, "SUMMARY NOT IMPORTANT"
    # 1) If there's already an abstract, use it directly
    if abstract := entry.get('abstract'):
        log_thought(f"Used cached abstract for {src}")
        return src, abstract

    # 2) Otherwise invoke the LLM on the scraped text
    text = entry.get('text', '')
    out = struct2_llm.invoke(
        f"""Summarize this document in 200–300 words in the answer field:
{text[:1000]}

Also, please include your chain-of-thought in the justification field (20–30 words).
also if the summary is not present or you cannot access please output false in the isSummary field"""
    )
    if(out.isSummary == False):
        #log_thought(f"NO summary for {src}")
        return src,"SUMMARY NOT IMPORTANT"
    log_thought(out.justification)
    return src, out.answer

def generate_subqueries(text: str, query: str, max_subs: int = 2) -> dict:
    prompt = (
        f"Extract up to {max_subs} subqueries from the text below in the answer field "
        f"that will help in studying the topic of {query} better, with justifications.\n\n"
        f"{text[:3000]}"
        f"Also, please include your chain-of-thought in the justification field (20–30 words)."
    )
    model_out = struct_llm.invoke(f"{prompt}\n{text}")
    if not model_out or model_out.justification is None:
        return {"subqueries": []}
    log_thought(model_out.justification)
    subqs = [s for s in model_out.answer if isinstance(s, str)][:max_subs]
    return {"subqueries": subqs}

async def deep_research(
    query: str,
    visited: Optional[Set[str]] = None,
    max_depth: int = 1
) -> Dict:
    if visited is None:
        visited = set()
    if query in visited or max_depth < 0:
        return {}
    visited.add(query)

    #schema_json = SourceSelection.model_dump_json(indent=2)
    prompt = f"""
For the research query: "{query}"
1. Decide whether a web search, an arXiv search, or a mixed approach is most appropriate.
2. Recommend how many web pages to scrape and how many arXiv papers to fetch (each between 0 and 5).

Please return a JSON matching to the Pydantic schema:
"""

    sel_out = select_llm.invoke(
    prompt + "\nAlso, include your chain-of-thought in the justification field."
)
    log_thought(sel_out.justification)

# 1) Fetch URLs / paper IDs using the LLM-recommended limits
    web_limit   = sel_out.web_limit
    arxiv_limit = sel_out.arxiv_limit

    web_urls = []
    if sel_out.search_preference in ("web", "mixed"):
        web_urls = search_web(query).get("urls", [])[:web_limit]

    paper_ids = []
    if sel_out.search_preference in ("arxiv", "mixed"):
        paper_ids = [
        r.entry_id for r in arxiv.Search(query=query, max_results=arxiv_limit).results()
    ]

    sources = web_urls + paper_ids

    # # 1) Gather sources
    # web_urls  = search_web(query).get('urls', [])[:5]
    # paper_ids = [r.entry_id for r in arxiv.Search(query=query, max_results=5).results()]
    # sources   = web_urls + paper_ids
    log_thought(f"For query '{query}', I will consult these sources: {sources}")

    # 2) Crawl & cache
    fetch_papers(sources)
    cache.load_cache()
    #data = load_cache()
    #CACHE.update(data)
#     summaries: Dict[str, str] = {}
#     for src in sources:
#         text = cache.CACHE.get(src, {}).get('text', '')
#         out = structured_llm.invoke(
#             f"""Summarize this document in 200–300 words:
# {text[:1000]}

# Also, please include your chain-of-thought in the justification field (20–30 words)."""
#         )
#         log_thought(out.justification)
#         summaries[src] = out.answer

#     # 3) Rank top sources with rationale
#     listing = "\n".join(f"- {s}: {summaries[s]}" for s in summaries)
    summaries: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = { pool.submit(summarize_src, src): src for src in sources }
        for future in as_completed(futures):
            src, answer = future.result()
            summaries[src] = answer

    listing = "\n".join(f"- {s}: {summaries[s]}" for s in summaries)
    try:
        rank_out = struct_llm.invoke(
            f"""Query: {query}
Candidates with summaries:
{listing}

Return the top 5 most relevant URLs in 'answer'.
Also, please include your chain-of-thought in the justification field (20–30 words)."""
        )
        log_thought(rank_out.justification)
        top_sources = rank_out.answer
    except Exception as e:
        log_thought(f"Ranking failed ({e}); defaulting to first 5.")
        top_sources = list(summaries)[:5]

    # 4) Combined summary + merge rationale
    combined = "\n\n".join(summaries[s] for s in top_sources if s in summaries)
    try:
        combined_out = structured_llm.invoke(
            f"""Give a combined summary of the following papers:

{combined}

Also, please include your chain-of-thought in the justification field (20–30 words)."""
        )
        log_thought(combined_out.justification)
        combined_summary = combined_out.answer
    except Exception:
        combined_summary = combined

    subqs = generate_subqueries(combined_summary, query ,max_subs=3)["subqueries"]


    # 6) Recurse
    node: Dict = {"query": query, "top_sources": top_sources, "subqueries": []}
    for sub in subqs:
        child = await deep_research(sub, visited=visited, max_depth=max_depth-1)
        if child:
            node["subqueries"].append(child)
    return node


async def deep_runner():
    
    print("About to call interactive_plan_loop…")
    query = input("Enter your research query: ").strip()

    try:
        final_plan = await interactive_plan_loop(query)
    except Exception as e:
        logging.error("Error during plan loop: %s", e)
        print("An error occurred while generating the plan. Please try again later.")
        return

    # Extract the tasks list from the plan
    tasks = final_plan.dict().get("tasks", [])
    if not tasks:
        print("No tasks were generated. Exiting.")
        return

    # Iterate over each task and call deep_research on its description
    result_tree = {
        "query": query,
        "tasks": []
    }
    
    if os.path.exists("cot.log"):
        os.remove("cot.log")

    visited=set()

    for task in tasks:
        task_id = task.get("id")
        description = task.get("description", "")
        print(f"Researching Task {task_id}: {description!r}")

        # You can pass a fresh `visited` set if your deep_research supports it
        node = await deep_research(
    query=description,
    max_depth=1,
    visited=visited
)

        # Attach the result under this task
        result_tree["tasks"].append({
            "id": task_id,
            "description": description,
            "result": node
        })
    # Pretty‐print the combined tree
    print("Finished deep_research for all tasks; result:")
    print(json.dumps(result_tree, indent=2))
    return result_tree