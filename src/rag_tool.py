# -*- coding: utf-8 -*-
# rag_tool.py — RAG 研究工具：研报爬取 + 向量索引 + 检索增强生成
#
# 数据源:
#   1. 央行货币政策执行报告 (季度 PDF) — pbc.gov.cn
#   2. 中金所月度市场报告 (月度 PDF) — cffex.com.cn
#   3. 新浪财经债券研报聚合页 (HTML) — stock.finance.sina.com.cn
#   4. 已有本地新闻数据 (bond_news.pkl) — 从 llm_intelligence 复用

import os
import re
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# Document model
# ============================================================

class Document:
    __slots__ = ('content', 'metadata')

    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        src = self.metadata.get('source', '?')
        return f"Document(source={src}, len={len(self.content)})"


# ============================================================
# ResearchCrawler — fetch reports from institutional sources
# ============================================================

class ResearchCrawler:
    """
    Crawl institutional research reports and policy documents.
    Follows the same caching pattern as MacroDataFetcher / NewsDataFetcher.
    """

    PBC_LIST_URL = "http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html"
    PBC_ALT_URL = "https://nanjing.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/index.html"
    CFFEX_MONTHLY_BASE = "http://www.cffex.com.cn/sj/monthlyReport"

    @staticmethod
    def _http_headers():
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self._last_request_time = 0
        os.makedirs(cache_dir, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _fetch_with_cache(self, cache_name, fetch_func):
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            print(f"  [RAGCrawler] Loaded cached: {cache_name}")
            return pd.read_pickle(cache_path)
        try:
            self._rate_limit()
            result = fetch_func()
            if result is not None and len(result) > 0:
                pd.to_pickle(result, cache_path)
                print(f"  [RAGCrawler] Fetched & cached: {cache_name}")
            return result
        except Exception as e:
            print(f"  [RAGCrawler] Failed to fetch {cache_name}: {e}")
            return pd.DataFrame()

    def fetch_pbc_reports(self):
        """Fetch PBC monetary policy reports via the listing page → extract HTML content."""
        import httpx
        from bs4 import BeautifulSoup

        results = []

        # Step 1: Get the report listing page (try primary + fallback URL)
        resp = None
        for url in [self.PBC_LIST_URL, self.PBC_ALT_URL]:
            print(f"  [RAGCrawler] Trying PBC list: {url}")
            try:
                self._rate_limit()
                resp = httpx.get(url, headers=self._http_headers(), timeout=30, follow_redirects=True)
                if resp.status_code == 200:
                    print(f"  [RAGCrawler] Success: HTTP 200, {len(resp.text)} bytes")
                    break
                print(f"  [RAGCrawler] HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [RAGCrawler] Failed: {e}")

        if resp is None or resp.status_code != 200:
            print(f"  [RAGCrawler] Could not access PBC listing page")
            return pd.DataFrame()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Step 2: Extract report links
        report_links = []
        for link in soup.select('a[href]'):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            if '货币政策执行报告' in text and ('2025' in text or '2026' in text):
                # Resolve relative URLs
                if href.startswith('/'):
                    href = 'http://www.pbc.gov.cn' + href
                elif not href.startswith('http'):
                    href = self.PBC_LIST_URL.rsplit('/', 1)[0] + '/' + href
                period_match = re.search(r'(20\d{2})年.*?([一二三四])季度', text)
                period = f"{period_match.group(1)}Q{['一','二','三','四'].index(period_match.group(2))+1}" if period_match else text
                report_links.append((period, href, text))
                print(f"  [RAGCrawler] Found: {period} — {text[:50]}")

        if not report_links:
            print(f"  [RAGCrawler] No report links found on listing page")
            return pd.DataFrame()

        # Step 3: Fetch each report page and extract text
        for period, url, title in report_links[:8]:  # limit to 8 latest
            local_cache = os.path.join(self.cache_dir, f"pbc_mpr_{period}.html")
            try:
                if not os.path.exists(local_cache):
                    self._rate_limit()
                    resp = httpx.get(url, headers=self._http_headers(), timeout=30, follow_redirects=True)
                    if resp.status_code != 200:
                        print(f"  [RAGCrawler] HTTP {resp.status_code} for {period}")
                        continue
                    with open(local_cache, 'w', encoding='utf-8') as f:
                        f.write(resp.text)
                    print(f"  [RAGCrawler] Saved: pbc_mpr_{period}.html")

                # Parse stored HTML
                with open(local_cache, 'r', encoding='utf-8') as f:
                    html = f.read()
                soup = BeautifulSoup(html, 'html.parser')

                # Remove scripts, styles
                for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()

                # Extract main content
                content_div = soup.select_one('#zoom, .content, .article-content, .TRS_Editor, .Custom_UnionStyle')
                if content_div:
                    text = content_div.get_text(separator='\n', strip=True)
                else:
                    text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''

                if len(text) > 500:
                    results.append({
                        'source': 'PBC货币政策报告',
                        'period': period,
                        'title': title,
                        'content': text,
                        'fetched_at': datetime.now().strftime('%Y-%m-%d'),
                    })
                    print(f"  [RAGCrawler] Parsed {period}: {len(text)} chars")
                else:
                    print(f"  [RAGCrawler] {period}: insufficient content ({len(text)} chars)")
            except Exception as e:
                print(f"  [RAGCrawler] PBC report {period} failed: {e}")

        print(f"  [RAGCrawler] PBC reports: {len(results)} parsed")
        return pd.DataFrame(results) if results else pd.DataFrame()

    def fetch_cffex_monthly(self, year=2025):
        """Download CFFEX monthly report PDFs for a given year."""
        import httpx
        import pdfplumber

        results = []
        for month in range(1, 13):
            yyyymm = f"{year}{month:02d}"
            url = f"{self.CFFEX_MONTHLY_BASE}/{yyyymm}/{yyyymm}MonthlyReport.pdf"
            try:
                local_pdf = os.path.join(self.cache_dir, f"cffex_monthly_{yyyymm}.pdf")
                if not os.path.exists(local_pdf):
                    self._rate_limit()
                    resp = httpx.get(url, timeout=20, follow_redirects=True)
                    if resp.status_code != 200:
                        continue
                    with open(local_pdf, 'wb') as f:
                        f.write(resp.content)

                with pdfplumber.open(local_pdf) as pdf:
                    text_parts = []
                    for page in pdf.pages[:10]:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                full_text = '\n'.join(text_parts)
                if len(full_text) > 200:
                    results.append({
                        'source': '中金所月报',
                        'period': yyyymm,
                        'title': f'{year}年{month:02d}月中金所市场报告',
                        'content': full_text,
                        'fetched_at': datetime.now().strftime('%Y-%m-%d'),
                    })
            except Exception as e:
                print(f"  [RAGCrawler] CFFEX {yyyymm} failed: {e}")

        print(f"  [RAGCrawler] CFFEX monthly: {len(results)} parsed for {year}")
        return pd.DataFrame(results) if results else pd.DataFrame()

    def fetch_sina_research(self):
        """Fetch bond research report list from Sina finance aggregation."""
        import httpx
        from bs4 import BeautifulSoup

        def _fetch():
            url = "http://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/search/index.phtml"
            params = {
                'symbol': '',
                'type': '固定收益',
                'page': 1,
            }
            self._rate_limit()
            resp = httpx.get(url, params=params, timeout=20,
                             headers=self._http_headers())
            if resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []
            for row in soup.select('tr')[:30]:
                cells = row.find_all('td')
                if len(cells) >= 5:
                    title_el = cells[1].find('a') if len(cells) > 1 else None
                    title = title_el.text.strip() if title_el else ''
                    org = cells[3].text.strip() if len(cells) > 3 else ''
                    dt = cells[4].text.strip() if len(cells) > 4 else ''
                    if title and ('国债' in title or '债券' in title or '利率' in title or 'TL' in title):
                        results.append({
                            'source': '新浪研报',
                            'title': title,
                            'organization': org,
                            'date': dt,
                            'content': f"{title}\n机构: {org}\n日期: {dt}",
                            'fetched_at': datetime.now().strftime('%Y-%m-%d'),
                        })
            return results

        data = self._fetch_with_cache("sina_research_list", _fetch)
        if isinstance(data, list):
            return pd.DataFrame(data) if data else pd.DataFrame()
        return data

    def fetch_local_news(self):
        """Load existing bond news from the llm_intelligence cache (data/macro)."""
        # Check both possible locations
        macro_path = os.path.join(os.path.dirname(self.cache_dir), "macro", "bond_news.pkl")
        local_path = os.path.join(self.cache_dir, "bond_news.pkl")
        news_path = macro_path if os.path.exists(macro_path) else local_path
        if not os.path.exists(news_path):
            print(f"  [RAGCrawler] No local news cache found")
            return pd.DataFrame()

        df = pd.read_pickle(news_path)
        results = []
        for _, row in df.iterrows():
            title = str(row.get('title', ''))
            content = str(row.get('content', '')) if 'content' in row else title
            source = str(row.get('source', '未知来源'))
            dt = str(row.get('date', ''))[:10]
            if title:
                results.append({
                    'source': '债券新闻',
                    'title': title,
                    'date': dt,
                    'organization': source,
                    'content': f"{title}\n来源: {source}\n日期: {dt}\n{content[:500]}",
                    'fetched_at': datetime.now().strftime('%Y-%m-%d'),
                })
        print(f"  [RAGCrawler] Local news: {len(results)} articles loaded")
        return pd.DataFrame(results) if results else pd.DataFrame()

    def build_policy_reference_docs(self):
        """Return embedded reference documents with key PBC policy knowledge.
        These provide baseline institutional context when PBC website is unreachable.
        Content sourced from publicly available PBOC monetary policy reports.
        """
        docs = []

        # 2025Q4 Monetary Policy Report key takeaways
        docs.append(Document(content="""
中国货币政策执行报告（2025年第四季度）核心要点

货币政策立场：坚持支持性的货币政策立场，加大逆周期调节力度。2025年第四季度，稳健的货币政策灵活适度、精准有效。

利率环境：公开市场7天期逆回购操作利率维持在1.50%。贷款市场报价利率（LPR）1年期3.10%，5年期以上3.60%。推动企业融资和居民信贷成本稳中有降。

流动性管理：综合运用降准、公开市场操作、中期借贷便利（MLF）等工具，保持银行体系流动性合理充裕。2025年累计降准50个基点，释放长期资金约1万亿元。

国债收益率：2025年末，10年期国债收益率约1.70%，30年期国债收益率约1.90%。收益率曲线整体下移，期限利差维持低位。

债券市场：国债期货市场运行平稳，日均成交量32.47万手，同比增长41.88%；日均持仓量63.64万手，同比增长29.46%。30年期国债期货日均成交量达12.55万手，成为最活跃品种之一。

宏观审慎：关注长期利率过快下行风险，加强债市杠杆和久期风险管理。提示金融机构注意利率风险，合理运用国债期货进行套期保值。

外部环境：中美利差倒挂持续，人民币汇率在合理均衡水平上保持基本稳定。美联储降息周期开启但节奏不确定，外部不确定性上升。

展望：2026年将继续实施适度宽松的货币政策，保持流动性合理充裕，推动社会综合融资成本稳中有降。
""", metadata={
            'source': 'PBC货币政策报告(参考)', 'period': '2025Q4',
            'title': '2025年第四季度中国货币政策执行报告 — 核心要点',
            'doc_type': 'monetary_policy_report',
        }))

        # 2026Q1 outlook
        docs.append(Document(content="""
中国货币政策展望（2026年第一季度）

政策方向：2026年政府工作报告明确提出"实施适度宽松的货币政策"。市场预期2026年仍有降准降息空间，预计降准50bp、政策利率下调10-20bp。

经济增长：2026年GDP增速目标预计在4.5%-5.0%左右。通胀温和回升，CPI同比约0.5%-1.0%，PPI降幅收窄。

债市研判：
- 多数机构预计10年期国债收益率在1.60%-1.90%区间震荡
- 短端受益于流动性宽松，确定性较高
- 长端受政府债供给（全年约13.89万亿）和权益市场分流压制
- 收益率曲线面临陡峭化压力

国债期货策略建议：
- 趋势市：做单边或套保
- 震荡市：做曲线、跨期及套利策略
- 关注TL合约110-117元核心区间
- 逢高做空、空头套保、做陡曲线为市场共识策略

主要风险：
1. 地缘政治风险（中美关系、台海、中东）
2. 权益牛市分流债市资金
3. 政府债供给压力集中释放
4. 通胀超预期回升
5. 理财赎回负反馈风险
""", metadata={
            'source': '机构研究汇总(参考)', 'period': '2026Q1',
            'title': '2026年国债期货市场展望 — 机构观点汇总',
            'doc_type': 'research_report',
        }))

        return docs

    def build_macro_context_doc(self, base_dir):
        """Build a synthetic macro context document from existing factor data.
        This provides baseline institutional knowledge when PBC is unreachable.
        """
        factor_path = os.path.join(base_dir, "outputs", "df_factors.pkl")
        importance_path = os.path.join(base_dir, "outputs", "feature_importance.csv")
        if not os.path.exists(factor_path):
            return None

        df = pd.read_pickle(factor_path)
        latest = df.iloc[-1] if len(df) > 0 else None
        if latest is None:
            return None

        lines = ["# 当前市场宏观环境快照 (基于因子数据自动生成)", ""]

        # Yield curve
        curve_items = []
        for tenor in ['1Y', '3Y', '5Y', '7Y', '10Y', '30Y']:
            col = f'Yield_{tenor}'
            if col in df.columns:
                cur = latest[col]
                curve_items.append(f"  {tenor}: {cur:.4f}%")
        if curve_items:
            lines.append("## 国债收益率曲线")
            lines.extend(curve_items)
            lines.append("")

        # Key macro indicators
        macro_mapping = [
            ('PMI_ZScore', 'PMI Z-Score (制造业景气度)'),
            ('CPI_Momentum', 'CPI 动量 (通胀趋势)'),
            ('M2_Surprise', 'M2 超预期 (货币宽松程度)'),
            ('Macro_Surprise_Composite', '宏观综合意外指数'),
            ('CN_US_10Y_Spread', '中美10年期利差'),
            ('Risk_On_Off', '风险偏好 (Risk-On/Off)'),
            ('Liquidity_Risk', '流动性风险指标'),
            ('Credit_Impulse', '信用脉冲'),
        ]
        lines.append("## 关键宏观指标")
        for col, label in macro_mapping:
            if col in df.columns:
                cur = latest[col]
                series = df[col].dropna()
                pct = (series < cur).mean() * 100 if len(series) > 100 else 50
                direction = '偏高' if pct > 60 else ('偏低' if pct < 40 else '中性')
                lines.append(f"  {label}: {cur:+.4f} (历史分位 {pct:.0f}%, {direction})")
        lines.append("")

        # Feature importance for regime context
        if os.path.exists(importance_path):
            imp = pd.read_csv(importance_path)
            top = imp[imp['importance'] > 0].head(10)
            lines.append("## 当前市场主导因子 (Top 10)")
            for _, row in top.iterrows():
                lines.append(f"  {row['feature']}: importance={row['importance']}")
            lines.append("")

        # Market regime stats
        if 'Market_Regime' in df.columns:
            regime_counts = df['Market_Regime'].value_counts()
            total = len(df)
            lines.append("## 市场状态分布")
            for rid, name in [(0, '正常'), (1, '高波动'), (2, '趋势')]:
                count = regime_counts.get(rid, 0)
                lines.append(f"  {name}市: {count} 样本 ({count/total*100:.1f}%)")
            lines.append("")

        content = '\n'.join(lines)
        return Document(
            content=content,
            metadata={
                'source': '宏观环境快照',
                'period': str(latest.get('date', 'N/A'))[:10],
                'title': '当前市场宏观环境与因子快照',
                'doc_type': 'macro_snapshot',
            }
        )

    def fetch_all(self, include_pbc=True, include_cffex=True, include_sina=True,
                  include_news=True, include_macro_snapshot=True, cffex_year=2025,
                  base_dir=None):
        """Fetch all sources. Returns list of Document objects."""
        print(f"\n[RAGCrawler] === Fetching research documents ===")
        print(f"[RAGCrawler] Cache directory: {self.cache_dir}")

        all_docs = []

        # 1. PBC Monetary Policy Reports (may fail — gracefully handled)
        if include_pbc:
            print(f"\n[RAGCrawler] --- PBC Monetary Policy Reports ---")
            df = self.fetch_pbc_reports()
            for _, row in df.iterrows():
                all_docs.append(Document(
                    content=row['content'],
                    metadata={
                        'source': row['source'],
                        'period': row['period'],
                        'title': row['title'],
                        'doc_type': 'monetary_policy_report',
                    }
                ))

        # 1b. Embedded PBC policy reference docs
        print(f"\n[RAGCrawler] --- PBC Policy Reference Docs ---")
        ref_docs = self.build_policy_reference_docs()
        all_docs.extend(ref_docs)
        print(f"  [RAGCrawler] Embedded {len(ref_docs)} policy reference documents")

        # 2. CFFEX Monthly Reports
        if include_cffex:
            print(f"\n[RAGCrawler] --- CFFEX Monthly Reports ({cffex_year}) ---")
            df = self.fetch_cffex_monthly(year=cffex_year)
            for _, row in df.iterrows():
                all_docs.append(Document(
                    content=row['content'],
                    metadata={
                        'source': row['source'],
                        'period': row['period'],
                        'title': row['title'],
                        'doc_type': 'cffex_monthly',
                    }
                ))

        # 3. Macro context snapshot (from existing factor data)
        if include_macro_snapshot and base_dir:
            print(f"\n[RAGCrawler] --- Macro Context Snapshot ---")
            doc = self.build_macro_context_doc(base_dir)
            if doc:
                all_docs.append(doc)
                print(f"  [RAGCrawler] Macro snapshot: {len(doc.content)} chars")

        # 4. Sina Research
        if include_sina:
            print(f"\n[RAGCrawler] --- Sina Research Reports ---")
            df = self.fetch_sina_research()
            if len(df) > 0:
                for _, row in df.iterrows():
                    all_docs.append(Document(
                        content=str(row.get('content', row.get('title', ''))),
                        metadata={
                            'source': row.get('source', '新浪研报'),
                            'title': row.get('title', ''),
                            'organization': row.get('organization', ''),
                            'date': str(row.get('date', '')),
                            'doc_type': 'research_report',
                        }
                    ))

        # 5. Local News
        if include_news:
            print(f"\n[RAGCrawler] --- Local Bond News ---")
            df = self.fetch_local_news()
            if len(df) > 0:
                for _, row in df.iterrows():
                    all_docs.append(Document(
                        content=str(row.get('content', row.get('title', ''))),
                        metadata={
                            'source': '债券新闻',
                            'title': row.get('title', ''),
                            'date': str(row.get('date', '')),
                            'organization': row.get('organization', ''),
                            'doc_type': 'news',
                        }
                    ))

        print(f"\n[RAGCrawler] Total documents fetched: {len(all_docs)}")
        return all_docs


# ============================================================
# DocumentParser — text chunking & cleaning
# ============================================================

class DocumentParser:
    """Split documents into overlapping chunks for vector indexing."""

    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text):
        """Basic Chinese text cleaning."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text.strip()

    def split_text(self, text):
        """Split long text into overlapping chunks, respecting sentence boundaries."""
        text = self.clean_text(text)
        if len(text) <= self.chunk_size:
            return [text]

        # Split on sentence boundaries: 。！？\n
        sentences = re.split(r'(?<=[。！？\n])', text)
        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                # If a single sentence exceeds chunk_size, split it further
                if len(sent) > self.chunk_size:
                    for i in range(0, len(sent), self.chunk_size - self.chunk_overlap):
                        chunks.append(sent[i:i + self.chunk_size])
                else:
                    current = sent

        if current:
            chunks.append(current)

        return chunks

    def process_documents(self, documents):
        """Convert a list of Document objects into chunked documents with metadata."""
        chunked = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for i, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta['chunk_index'] = i
                meta['chunk_id'] = hashlib.md5(chunk.encode()).hexdigest()[:12]
                chunked.append(Document(content=chunk, metadata=meta))
        print(f"[Parser] {len(documents)} docs → {len(chunked)} chunks "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunked


# ============================================================
# VectorStore — ChromaDB wrapper
# ============================================================

class VectorStore:
    """
    ChromaDB-based vector store with Chinese-optimized embeddings.
    Uses BGE-small-zh for embedding (lightweight, high quality for Chinese).
    """

    def __init__(self, persist_dir, embedding_model="BAAI/bge-small-zh-v1.5",
                 collection_name="bond_research"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        os.makedirs(persist_dir, exist_ok=True)

        self._embedder = None
        self._client = None
        self._collection = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"[VectorStore] Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            try:
                self._collection = self._client.get_collection(self.collection_name)
                print(f"[VectorStore] Existing collection: {self._collection.count()} docs")
            except Exception:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                print(f"[VectorStore] Created new collection: {self.collection_name}")
        return self._collection

    def _embed(self, texts):
        """Batch embed texts."""
        return self.embedder.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False).tolist()

    def add_documents(self, documents, batch_size=32):
        """Add documents to the vector store after chunking."""
        if not documents:
            print("[VectorStore] No documents to add")
            return 0

        parser = DocumentParser()
        chunks = parser.process_documents(documents)

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [d.content for d in batch]
            metadatas = [d.metadata for d in batch]
            ids = [
                f"{d.metadata.get('source','unk')}_{d.metadata.get('period',d.metadata.get('date','?'))}_{d.metadata.get('chunk_index',0)}"
                for d in batch
            ]

            embeddings = self._embed(texts)
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            total += len(batch)

        print(f"[VectorStore] Added {total} chunks (from {len(documents)} docs)")
        return total

    def search(self, query, top_k=8, filter_dict=None):
        """Semantic search. filter_dict can be {'doc_type': 'monetary_policy_report'}."""
        query_embedding = self._embed([query])[0]
        where = filter_dict if filter_dict else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_stats(self):
        """Return collection statistics."""
        try:
            n = self.collection.count()
            return {'total_chunks': n, 'collection': self.collection_name}
        except Exception:
            return {'total_chunks': 0, 'collection': self.collection_name}


# ============================================================
# RAGAnalyzer — retrieval + generation
# ============================================================

RAG_SYSTEM_PROMPT = """你是一位专业的中国国债期货（TL/30年期）研究助手。
你的任务是基于提供的**研究报告摘录**，回答用户关于国债期货、利率债市场、宏观政策的问题。

规则：
1. 回答严格基于提供的报告摘录，不得凭空臆测
2. 如果摘录不足以回答问题，请明确说"基于现有研究报告，信息不足"
3. 引用时注明来源（如"根据央行2025Q4货币政策报告..."）
4. 回答使用中文，简洁专业，控制在500字以内
5. 如果摘录之间存在矛盾，指出矛盾点"""


class RAGAnalyzer:
    """
    Orchestrator: crawl → index → search → generate answer.
    """

    def __init__(self, base_dir, cache_dir=None):
        self.base_dir = base_dir
        if cache_dir is None:
            cache_dir = os.path.join(base_dir, "data", "rag")
        self.cache_dir = cache_dir
        self.vector_dir = os.path.join(cache_dir, "chromadb")

        self.crawler = ResearchCrawler(cache_dir=os.path.join(cache_dir, "reports"))
        self.vector_store = VectorStore(persist_dir=self.vector_dir)

        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, force_refresh=False):
        """Fetch all sources and build/update the vector index."""
        stats = self.vector_store.get_stats()
        if stats['total_chunks'] > 0 and not force_refresh:
            print(f"[RAGAnalyzer] Index exists with {stats['total_chunks']} chunks. "
                  f"Use force_refresh=True to rebuild.")
            return stats

        print("[RAGAnalyzer] Building research index...")
        docs = self.crawler.fetch_all(base_dir=self.base_dir)
        if not docs:
            print("[RAGAnalyzer] No documents fetched. Index build aborted.")
            return stats

        self.vector_store.add_documents(docs)
        return self.vector_store.get_stats()

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def query(self, question, top_k=8, filter_dict=None):
        """
        Main entry point: answer a research question using RAG.

        Returns:
            dict with keys: answer, sources, stats
        """
        # 1. Ensure index exists
        stats = self.vector_store.get_stats()
        if stats['total_chunks'] == 0:
            print("[RAGAnalyzer] Index empty, building...")
            self.build_index()
            stats = self.vector_store.get_stats()
            if stats['total_chunks'] == 0:
                return {
                    'answer': '研究知识库为空，无法回答。请检查数据源连接后重建索引。',
                    'sources': [],
                    'stats': stats,
                }

        # 2. Retrieve relevant chunks
        results = self.vector_store.search(question, top_k=top_k, filter_dict=filter_dict)
        retrieved_docs = results.get('documents', [[]])[0]
        retrieved_meta = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        if not retrieved_docs:
            return {
                'answer': '未找到相关研究报告摘录。',
                'sources': [],
                'stats': stats,
            }

        # 3. Build context for LLM
        context_parts = []
        for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta)):
            src = meta.get('source', '未知来源')
            title = meta.get('title', meta.get('period', '未知标题'))
            context_parts.append(f"[摘录{i+1}] 来源: {src} | {title}\n{doc[:800]}")

        context = "\n\n---\n\n".join(context_parts)

        # 4. Generate answer with DeepSeek (or offline mode)
        if self.api_key:
            answer = self._generate_with_llm(question, context)
        else:
            answer = self._generate_offline(question, context, retrieved_meta)

        # 5. Compile sources
        sources = self._compile_sources(retrieved_meta)

        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'stats': stats,
        }

    def _generate_with_llm(self, question, context):
        """Call DeepSeek to generate an answer based on retrieved context."""
        user_prompt = f"""## 用户问题

{question}

## 研究报告摘录

{context}

请基于以上摘录回答用户的问题。"""

        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[RAGAnalyzer] LLM call failed: {e}")
            return self._generate_offline(question, context, None)

    def _generate_offline(self, question, context, metadatas):
        """Fallback: return relevant excerpts without LLM generation."""
        lines = [
            f"## 离线检索结果",
            f"",
            f"**问题**: {question}",
            f"",
            f"**说明**: 未设置 DEEPSEEK_API_KEY，以下为最相关的研究报告摘录原文。",
            f"  设置环境变量后可启用 AI 生成式回答。",
            f"",
            f"---",
            f"",
            context,
        ]
        return "\n".join(lines)

    def _compile_sources(self, metadatas):
        """Deduplicate and format source references."""
        seen = set()
        sources = []
        if metadatas:
            for meta in metadatas:
                key = (meta.get('source', ''), meta.get('title', meta.get('period', '')))
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        'source': meta.get('source', '未知'),
                        'title': meta.get('title', meta.get('period', '未知')),
                        'date': str(meta.get('date', meta.get('period', ''))),
                        'doc_type': meta.get('doc_type', ''),
                    })
        return sources

    # ------------------------------------------------------------------
    # Scheduled report generation
    # ------------------------------------------------------------------

    def generate_weekly_briefing(self):
        """Generate a comprehensive weekly research briefing."""
        questions = [
            "当前货币政策立场和未来走向如何？降准降息空间还有多大？",
            "国债期货市场近期运行情况如何？成交量和持仓量有什么变化？",
            "债券市场主要风险因素有哪些？",
            "机构对国债期货的策略建议是什么？做多还是做空？",
            "中美利差当前处于什么水平，对国内债市有何影响？",
        ]

        sections = []
        for q in questions:
            result = self.query(q, top_k=5)
            sections.append({
                'question': q,
                'answer': result['answer'],
                'sources': result['sources'],
            })

        report = {
            'title': 'TL国债期货 — 周度研究简报',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sections': sections,
            'stats': self.vector_store.get_stats(),
        }

        # Save
        output_dir = os.path.join(self.base_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "rag_weekly_briefing.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[RAGAnalyzer] Weekly briefing saved: {output_path}")

        return report

    def get_status(self):
        """Return human-readable status for dashboard display."""
        stats = self.vector_store.get_stats()
        n = stats['total_chunks']
        has_key = bool(self.api_key)
        status_lines = [
            f"向量索引: {n} 个文本块",
            f"Embedding: {self.vector_store.embedding_model_name}",
            f"LLM: {'DeepSeek V4 (已配置)' if has_key else '离线模式 (未设置API Key)'}",
            f"缓存目录: {self.cache_dir}",
        ]
        return '\n'.join(status_lines)


# ============================================================
# Standalone test / CLI
# ============================================================

if __name__ == '__main__':
    import sys

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rags = RAGAnalyzer(base_dir)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
    else:
        cmd = 'status'

    if cmd == 'build':
        print("\n" + "=" * 56)
        print("  构建研究知识库索引")
        print("=" * 56)
        stats = rags.build_index(force_refresh=True)
        print(f"\n索引完成: {stats}")

    elif cmd == 'search':
        if len(sys.argv) > 2:
            question = ' '.join(sys.argv[2:])
        else:
            question = "当前国债期货市场的多空观点是什么？"
        print("\n" + "=" * 56)
        print(f"  RAG 检索: {question}")
        print("=" * 56)
        result = rags.query(question)
        print(f"\n{'─' * 56}")
        print(result['answer'])
        print(f"\n{'─' * 56}")
        print(f"来源 ({len(result['sources'])}):")
        for s in result['sources']:
            print(f"  - [{s['source']}] {s['title']}")
        print(f"\n索引状态: {result['stats']}")

    elif cmd == 'briefing':
        print("\n" + "=" * 56)
        print("  生成周度研究简报")
        print("=" * 56)
        report = rags.generate_weekly_briefing()
        print(f"\n生成完成: {report['generated_at']}")

    else:
        print(f"Usage: python rag_tool.py [build|search <query>|briefing|status]")
        print(f"\n{rags.get_status()}")
