import time
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from io import BytesIO
import arxiv
from config import settings

session = requests.Session()
CACHE: dict[str, str] = {}
def search_web(query: str, num_results: int = 5, retries: int = 5, delay: float = 5.0) -> dict[str, list[str]]:
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key':       settings.GOOGLE_API_KEY,
        'cx':        settings.GOOGLE_CX,
        'q':         query,
        'num':       num_results
    }
    for attempt in range(1, retries+1):
        try:
            resp = session.get(url, params=params, timeout=(5,20))
            resp.raise_for_status()
            items = resp.json().get('items', [])
            return {"urls": [item.get('link') for item in items]}
        except Exception as e:
            print(f"[WARN] search_web attempt {attempt} failed: {e}")
            if attempt < retries: time.sleep(delay)
    print(f"[ERROR] search_web giving up after {retries} attempts.")
    return {"urls": []}

def fetch_url(url: str, retries: int = 5, delay: float = 5.0) -> dict[str, str]:
    for attempt in range(1, retries+1):
        try:
            resp = session.get(url, timeout=(5,20))
            resp.raise_for_status()
            paras = [p.get_text() for p in BeautifulSoup(resp.text, 'html.parser').find_all('p')]
            return {'text': "\n".join(paras)[:2000]}
        except Exception as e:
            print(f"[WARN] fetch_url attempt {attempt} for {url} failed: {e}")
            if attempt < retries: time.sleep(delay)
    return {'text': ''}

def fetch_pdf(url: str, retries: int = 5, delay: float = 5.0) -> dict[str, str]:
    for attempt in range(1, retries+1):
        try:
            resp = session.get(url, timeout=(5,40))
            resp.raise_for_status()
            reader = PdfReader(BytesIO(resp.content))
            pages = [page.extract_text() or '' for page in reader.pages[:5]]
            return {'text': '\n'.join(pages)}
        except Exception as e:
            print(f"[WARN] fetch_pdf attempt {attempt} for {url} failed: {e}")
            if attempt < retries: time.sleep(delay)
    return {'text': ''}

def fetch_arxiv(paper_id: str) -> dict:
    paper = next(arxiv.Search(id_list=[paper_id]).results(), None)
    if not paper:
        return {'error': 'Not found'}
    return {
        'title': paper.title,
        'authors': [a.name for a in paper.authors],
        'abstract': paper.summary
    }
