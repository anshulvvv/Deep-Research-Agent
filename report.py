
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Annoy
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import cache
from config import settings
import os
import uuid

def random_filename(extension: str = "") -> str:
    # uuid4 gives a random 128-bit GUID
    name = uuid.uuid4().hex
    if extension and not extension.startswith("."):
        extension = "." + extension
    return name + extension


# ─── 1) Load the cache you saved in deep_runner ───────────────────────────────
#CACHE: Dict[str, Any] = {}




# ─── 2) Pydantic models ───────────────────────────────────────────────────────
class Section(BaseModel):
    heading: str
    subheadings: List[str] = Field(..., description="Bullet-point subtopics")

class TaskStructure(BaseModel):
    task_id: int
    sections: List[Section]

class SectionOutput(BaseModel):
    heading: str
    content: str

class TaskReportOut(BaseModel):
    task_id: int
    sections: List[SectionOutput]

# ─── 3) Helpers to flatten a node and gather all subqueries ───────────────────
def flatten_node(node: Dict) -> List[str]:
    texts = [
        f"Query: {node['query']}\n"
        f"Sources: {', '.join(node.get('top_sources', []))}\n"
    ]
    for src in node.get('top_sources', []):
        entry = cache.CACHE.get(src, {})
        # your cache entries from spider are dicts with 'text'
        text = entry.get("text") if isinstance(entry, dict) else entry
        texts.append(text or "")
    for child in node.get('subqueries', []):
        texts += flatten_node(child)
    return texts

def gather_subqueries(node: Dict) -> List[str]:
    subs = [node["query"]]
    for child in node.get("subqueries", []):
        subs += gather_subqueries(child)
    return subs

# ─── 4) LLM setup ─────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=settings.MODEL,
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.2,
    max_output_tokens=1024
)
struct_llm = llm.with_structured_output(TaskStructure)
report_llm = llm.with_structured_output(TaskReportOut)

# ─── 5) Build an Annoy-based RAG retriever ────────────────────────────────────
def build_rag_for_node(node: Dict, k: int = 5) -> RetrievalQA:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY
    )
    docs = [
        Document(page_content=t, metadata={})
        for t in flatten_node(node) if t and t.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)

    vect = Annoy.from_documents(
        chunks,
        embeddings,
        search_kwargs={"n_trees": 10}
    )
    retriever = vect.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

# ─── 6) Main report generation function ───────────────────────────────────────
def generate_reports(result_tree: Dict):
    final_reports: Dict[int, TaskReportOut] = {}
    task_info: List[Dict] = []
    cache.load_cache()
    # 6a) Extract task details & subqueries
    for task in result_tree["tasks"]:
        tid   = task["id"]
        desc  = task["description"]
        node  = task["result"]
        subqs = gather_subqueries(node)
        task_info.append({
            "id": tid,
            "description": desc,
            "subqueries": subqs
        })

    # 6b) Build the combined outline prompt
    lines = [
        "You are an expert academic report architect. Your job is to turn a set of tasks and sub-queries into a professional, publishable report outline.",
        "",
        "Below are the tasks and the sub-queries for each (with task id, description, and sub-queries):"
    ]
    for info in task_info:
        lines.append(f"Task {info['id']}: {info['description']}")
        lines.append("Sub-queries:")
        for sq in info["subqueries"]:
            lines.append(f"  - {sq}")
        lines.append("")

    lines.append(
        "Produce a report **structure** with:\n"
        "- An **Introduction** (2–3 bullet points: scope, objectives, context).\n"
        "- One major **Section** per Task, each with:\n"
        "  - A concise, descriptive **heading** relevant for a report.\n"
        "  - 2–3 logically grouped **subheadings** corresponding to the sub-queries.\n"
        "- A **Conclusion** that summarizes the entire report and suggests further reading.\n"
    )

    combined_prompt = "\n".join(lines)
    outline = struct_llm.invoke(combined_prompt)
    #node = next(t["result"] for t in result_tree["tasks"] if t["id"] == tid)
    qa   = build_rag_for_node(node)
    sections_out = []
    unique_sources: List[str] = []
    def collect_sources(node: Dict[str, Any]):
        for src in node.get("top_sources", []):
            if src not in unique_sources:
                unique_sources.append(src)
        for child in node.get("subqueries", []):
            collect_sources(child)

    for task in result_tree["tasks"]:
        collect_sources(task["result"])   
    sources_list_str = "\n".join(f"{i+1}. {src}" for i, src in enumerate(unique_sources))
    # 6c) For each task, generate full section content via RAG QA
    for sec in outline.sections:
        heading  = sec.heading
        subheads = sec.subheadings
        prompt = f"""
You are an expert research summarizer. Write high-quality, fact-grounded prose.
Section title, don't print this title in the report : "{heading}".
Cover these subtopics: {subheads}
1. Write a cohesive ~500 to 700 word narrative under the given title ,the word limit is a recommendation , befare DO NOT HALLUCINATE to fill up the words ,the length of the content must be according to the topic only so do not HALLUCINATE, make paragraphs for the subtopics and write a report that is clear and accurate.
2. After every factual statement or paraphrase, include an inline citation in parentheses referring to the source ID exactly as given (e.g. `(Source Number)`). 
the following are the sources : {sources_list_str}
do not print the sources in the report , i'll print them at the end of the report myself 
Constraints:
- Use an academic, neutral tone.
- You may add sub-subsections if it improves clarity.
- Make sure to not write the heading in every sub-task.

""".strip()
        content = qa.invoke(prompt)
        #print(content)
        sections_out.append(SectionOutput(heading=heading, content=content["result"]))

    # 5d) Wrap into the final Pydantic report
        report = TaskReportOut(task_id=tid, sections=sections_out)
        final_reports[tid] = report
        
    for tid, rpt in final_reports.items():
        with open(f"Report.json", "w", encoding="utf-8") as f:
        # <-- use model_dump_json, which _does_ accept indent
            f.write(rpt.model_dump_json(indent=2))
    print("Structured reports written for tasks:", list(final_reports.keys()))


    INPUT_JSON = "Report.json"
    OUTPUT_MD   = "final_report.md"

    # 1) Load the structured JSON
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) Grab the optional task_id (for a title) and the sections list
    task_id = data.get("task_id")
    sections = data.get("sections", [])

    # 3) Write the .md
    with open(OUTPUT_MD, "w", encoding="utf-8") as md:
        # # Optional title
        # if task_id is not None:
        #     md.write(f"# Task {task_id} Structured Report\n\n")

        # Each section as an H2 + body
        for sec in sections:
            heading = sec.get("heading", "").strip()
            content = sec.get("content", "").strip()

            #md.write(f"## {heading}\n\n")
            md.write(f"{content}\n\n")
        
        if unique_sources:
            md.write("## References\n\n")
            for i, url in enumerate(unique_sources, 1):
                md.write(f"{i}. {url}\n")

    print(f"Wrote Markdown report to {os.path.abspath(OUTPUT_MD)}")
