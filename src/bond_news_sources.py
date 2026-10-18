"""Bounded public announcement collectors, with no browser/login bypass."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from intelligence_events import redact_sensitive_text, utc_now_iso
from sentiment_monitor import ProviderDocument, ProviderFetchResult


@dataclass(frozen=True)
class OfficialSource:
    url: str
    host: str
    link_selector: str
    body_selector: str


SOURCES = {
    "pbc_news": OfficialSource(
        "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html",
        "pbc.gov.cn", "a[istitle], .newslist_style a", "#zoom"),
    "pbc_omo": OfficialSource(
        "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html",
        "pbc.gov.cn", "a[istitle], .newslist_style a", "#zoom"),
    "mof_bonds": OfficialSource(
        "https://www.mof.gov.cn/gkml/bulinggonggao/tongzhitonggao/",
        "mof.gov.cn", ".xwbd_lianbolistfrcon li a", ".TRS_Editor"),
}


class OfficialBondNewsProvider:
    MAX_ARTICLES_PER_RUN = 10

    def __init__(self, name, *, get_html=None, sleep_fn=time.sleep):
        if name not in SOURCES:
            raise ValueError("unknown official source")
        self.name = name
        self.spec = SOURCES[name]
        self.get_html = get_html or self._get_html
        self.sleep_fn = sleep_fn

    def _get_html(self, url):
        # Reuse the existing redirect/DNS/size/MIME guard. The fetch method has
        # no cache/init dependency and does not execute or index downloaded text.
        from rag_tool import ResearchCrawler
        fetcher = ResearchCrawler.__new__(ResearchCrawler)
        response = fetcher._safe_fetch(
            url, allowed_hosts={self.spec.host}, timeout=(5, 10),
            headers={"User-Agent": "F-Agent-BondMonitor/1.0 (public research)"},
            max_bytes=2 * 1024 * 1024,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text

    def _article_url(self, href):
        parsed = urlsplit(urljoin(self.spec.url, href))
        host = (parsed.hostname or "").lower()
        if (parsed.scheme not in {"http", "https"} or parsed.username or parsed.password
                or parsed.port not in (None, 443, 80)
                or not (host == self.spec.host or host.endswith("." + self.spec.host))):
            raise ValueError("official source contains a non-allowlisted article URL")
        # MOF publishes HTTP links; fetch those resources using HTTPS only.
        return urlunsplit(("https", host, parsed.path, parsed.query, ""))

    def parse_index(self, html):
        soup = BeautifulSoup(html, "html.parser")
        links = soup.select(self.spec.link_selector)
        if not links:
            raise ValueError(f"{self.name} list parser found no article links; possible layout change/block page")
        entries, seen = [], set()
        for link in links[:100]:
            title = link.get("title") or link.get_text("", strip=True)
            if self.name == "mof_bonds" and not any(x in title for x in ("国债", "地方债", "政府债", "债券")):
                continue
            url = self._article_url(link.get("href", ""))
            if url in seen:
                continue
            parent = link.find_parent("li") or link.find_parent("td")
            date_text = parent.get_text(" ", strip=True) if parent else ""
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", date_text)
            if not dates:
                raise ValueError(f"{self.name} article listing has no publication date")
            published = datetime.strptime(dates[-1], "%Y-%m-%d").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            entries.append((url, title, published.isoformat()))
            seen.add(url)
        return entries

    def parse_article(self, html):
        soup = BeautifulSoup(html, "html.parser")
        body = soup.select_one(self.spec.body_selector)
        if body is None:
            raise ValueError(f"{self.name} article body missing; possible layout change/block page")
        for unwanted in body.select("script, style, iframe, form, noscript"):
            unwanted.decompose()
        # Preserve paragraph/table boundaries; concatenate inline spans so
        # typeset numbers such as '202' + '6' remain intact.
        for node in body.select("p, tr, br"):
            node.append("\n")
        text = body.get_text("", strip=False)
        text = re.sub(r"[ \t\r\u3000]+", "", text)
        text = re.sub(r"\n+", "\n", text).strip()
        if len(text) < 20:
            raise ValueError(f"{self.name} article body too short")
        return text

    def fetch_since(self, cursor):
        entries = self.parse_index(self.get_html(self.spec.url))
        state = {}
        if cursor:
            try:
                state = json.loads(cursor.cursor_value)
            except (ValueError, TypeError):
                pass
        seen = state.get("seen", []) if isinstance(state, dict) else []
        seen = [x for x in seen if isinstance(x, str)][:200]
        unseen = [entry for entry in entries if entry[0] not in seen]
        # Prioritize backlog, then rotate through the visible list for revisions.
        known = [entry for entry in entries if entry[0] in seen]
        offset = state.get("refresh_offset", 0) if isinstance(state, dict) else 0
        offset = offset % max(1, len(known)) if type(offset) is int else 0
        rotated = known[offset:] + known[:offset]
        candidates = unseen + rotated
        refreshed = max(0, min(len(rotated), self.MAX_ARTICLES_PER_RUN - len(unseen)))
        docs, errors = [], []
        for url, title, published_at in candidates[:self.MAX_ARTICLES_PER_RUN]:
            self.sleep_fn(1.0)
            try:
                content = self.parse_article(self.get_html(url))
            except Exception as exc:
                errors.append(redact_sensitive_text(f"article fetch failed: {type(exc).__name__}: {exc}")[:500])
                continue
            docs.append(ProviderDocument(url, title, content, url, published_at,
                                         {"publisher": self.spec.host, "publication_precision": "day",
                                          "list_url": self.spec.url}))
            seen = [url] + [item for item in seen if item != url]
        if len(unseen) > self.MAX_ARTICLES_PER_RUN:
            errors.append("bounded backlog: remaining visible articles will be fetched on next poll")
        return ProviderFetchResult(docs, json.dumps({"seen": seen[:200],
            "refresh_offset": (offset + refreshed) % max(1, len(known))}, sort_keys=True),
                                   utc_now_iso(), tuple(errors))


def configured_providers(config):
    from sentiment_monitor import EastmoneyTLNewsProvider
    return [EastmoneyTLNewsProvider(keywords=tuple(
                word for task in config.tasks for word in (*task.entities, *task.keywords)
            )) if name == "eastmoney_tl" else OfficialBondNewsProvider(name)
            for name in config.sources]
