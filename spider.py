import re
from io import BytesIO
import arxiv
import scrapy
from bs4 import BeautifulSoup
from w3lib.html import remove_tags, replace_entities
from PyPDF2 import PdfReader
from fake_useragent import UserAgent
from crochet import setup, wait_for
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from twisted.internet.defer import inlineCallbacks, returnValue, DeferredList
import cache
from typing import Any, Dict

setup()   # start Twisted reactor once

ARXIV_ABS_RE = re.compile(r'arxiv\.org/abs/([^/?]+)')

def clean_html(raw_html: str) -> str:
    # 1) remove scripts & styles
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # 2) remove any leftover tags/entities
    text = remove_tags(text)
    text = replace_entities(text)
    # 3) collapse whitespace
    return re.sub(r"\s+", " ", text).strip()

class PaperItem(scrapy.Item):
    src      = scrapy.Field()
    title    = scrapy.Field()
    authors  = scrapy.Field()
    abstract = scrapy.Field()
    text     = scrapy.Field()
    pdf_url  = scrapy.Field()

class RandomUserAgentMiddleware:
    def __init__(self):
        self.ua = UserAgent()
    @classmethod
    def from_crawler(cls, crawler):
        return cls()
    def process_request(self, request, spider):
        request.headers['User-Agent'] = self.ua.random

class RobustSpider(scrapy.Spider):
    name = 'robust_spider'
    custom_settings = {
        'LOG_ENABLED': False,
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 5,
        'AUTOTHROTTLE_ENABLED': True,
        'CONCURRENT_REQUESTS': 16,
        'DOWNLOAD_TIMEOUT': 20,
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
            f'{__name__}.RandomUserAgentMiddleware': 400,
        },
    }
    def __init__(self, start_urls, **kw):
        super().__init__(**kw)
        self.start_urls = start_urls

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, errback=self.errback, dont_filter=True)

    def parse(self, response):
        src = response.url
        if src in cache.CACHE:
            return

        item = PaperItem(src=src)
        m = ARXIV_ABS_RE.search(src)
        if m:
            # arXiv abstract case
            pid = m.group(1)
            paper = next(arxiv.Search(id_list=[pid]).results(), None)
            if paper:
                item['title']    = paper.title
                item['authors']  = [a.name for a in paper.authors]
                item['abstract'] = paper.summary
                yield scrapy.Request(
                    paper.pdf_url,
                    callback=self.parse_pdf,
                    meta={'item': item},
                    errback=self.errback
                )
                return

        if src.lower().endswith('.pdf'):
            # direct PDF link
            yield scrapy.Request(src, callback=self.parse_pdf, meta={'item': item}, errback=self.errback)
            return

        # fallback: scrape HTML page
        item['title'] = response.url
        # clean out all tags, scripts, entities, whitespace
        item['text']  = clean_html(response.text)
        cache.CACHE[src] = dict(item)
        yield item

    def parse_pdf(self, response):
        item = response.meta['item']
        reader = PdfReader(BytesIO(response.body))
        item['text']    = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        item['pdf_url'] = response.url
        cache.CACHE[item['src']] = dict(item)
        yield item

    def errback(self, failure):
        self.logger.warning(f"Fetch error {failure.request.url}: {failure.value}")

def _crawl_deferred(sources):
    runner = CrawlerRunner(settings=RobustSpider.custom_settings)

    # 1) launch one spider per URL, immediately
    deferreds = [
        runner.crawl(RobustSpider, start_urls=[url])
        for url in sources
    ]

    # 2) wait for all of them to finish (consumeFailures=True so one bad crawl
    #    won’t stop the whole batch)
    all_done = DeferredList(deferreds, consumeErrors=True)

    # 3) once everyone’s done, save and return the shared cache
    def _persist_and_forward(results):
        cache.save_cache()
        return cache.CACHE

    all_done.addCallback(_persist_and_forward)
    return all_done

@wait_for(timeout=300)
def fetch_papers(sources: list[str]) -> dict[str, dict]:
    cache.load_cache()
    configure_logging()
    deferred = _crawl_deferred(sources)
    def _persist_and_forward(result):
        cache.save_cache()
        return result
    deferred.addBoth(_persist_and_forward)
    return deferred
