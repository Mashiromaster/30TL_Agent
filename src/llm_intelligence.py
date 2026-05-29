# -*- coding: utf-8 -*-
# llm_intelligence.py — LLM-powered market intelligence for TL strategy

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime


# ============================================================
# NewsDataFetcher — Bond market news fetching with caching
# ============================================================

class NewsDataFetcher:
    """
    Fetch Chinese bond market news from financial data sources.
    Follows the same caching pattern as MacroDataFetcher in data_fetcher.py.
    """

    BOND_KEYWORDS = [
        '国债', 'TL', '债券', '利率', '央行', '收益率', '货币',
        'LPR', 'MLF', '逆回购', '降准', '降息', '债市', '国债期货',
        '公开市场', 'OMO', 'SLF', 'PSL', '流动性', '资金面',
        '中债', '信用债', '利率债', '地方债', '专项债',
    ]

    def __init__(self, cache_dir, start_date="20260401", end_date=None):
        self.cache_dir = cache_dir
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self._last_request_time = 0
        os.makedirs(cache_dir, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request_time = time.time()

    def _fetch_with_cache(self, cache_name, fetch_func):
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            print(f"  [NewsFetcher] Loaded cached: {cache_name}")
            return pd.read_pickle(cache_path)

        try:
            self._rate_limit()
            df = fetch_func()
            if df is not None and len(df) > 0:
                df.to_pickle(cache_path)
                print(f"  [NewsFetcher] Fetched & cached: {cache_name} ({len(df)} rows)")
            else:
                print(f"  [NewsFetcher] Empty result: {cache_name}")
            return df
        except Exception as e:
            print(f"  [NewsFetcher] Failed to fetch {cache_name}: {e}")
            return pd.DataFrame()

    def fetch_eastmoney_news(self):
        """Fetch financial news from 东方财富, filter for bond-related."""
        def _fetch():
            import akshare as ak
            df = ak.stock_news_em()
            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.copy()
            # Normalize column names (may vary by AKShare version)
            col_map = {}
            for c in df.columns:
                if '标题' in str(c) or 'title' in str(c).lower():
                    col_map[c] = 'title'
                elif '内容' in str(c) or 'content' in str(c).lower():
                    col_map[c] = 'content'
                elif '时间' in str(c) or 'date' in str(c).lower() or 'time' in str(c).lower():
                    col_map[c] = 'date'
                elif '来源' in str(c) or 'source' in str(c).lower():
                    col_map[c] = 'source'
                elif '链接' in str(c) or 'url' in str(c).lower():
                    col_map[c] = 'url'

            df = df.rename(columns=col_map)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

            # Filter for bond-related keywords
            if 'title' in df.columns:
                pattern = '|'.join(self.BOND_KEYWORDS)
                mask = df['title'].str.contains(pattern, case=False, na=False)
                if 'content' in df.columns:
                    mask |= df['content'].str.contains(pattern, case=False, na=False)
                df = df[mask]

            return df.reset_index(drop=True)

        return self._fetch_with_cache("bond_news", _fetch)

    def fetch_all(self):
        """Fetch all news sources. Returns dict of {name: DataFrame}."""
        print(f"[NewsFetcher] Fetching bond market news...")
        print(f"[NewsFetcher] Cache directory: {self.cache_dir}")

        data = {}
        sources = [
            ('bond_news', self.fetch_eastmoney_news),
        ]

        for name, func in sources:
            print(f"\n[NewsFetcher] === {name} ===")
            df = func()
            if df is not None and len(df) > 0:
                data[name] = df
            else:
                data[name] = pd.DataFrame()
                print(f"  [NewsFetcher] {name}: no data")

        success = sum(1 for v in data.values() if len(v) > 0)
        print(f"\n[NewsFetcher] Complete: {success}/{len(sources)} sources fetched")
        return data


# ============================================================
# LLMAnalyzer — DeepSeek V4 market intelligence analysis
# ============================================================

SYSTEM_PROMPT = """你是一位专业的中国国债期货（TL/30年期）量化策略分析师。
你的任务是基于提供的量化数据上下文和最新市场新闻，生成一份结构化、专业的市场情报分析报告。

要求：
1. 分析必须基于提供的数据和新闻，不可凭空臆测
2. 使用专业的固定收益分析术语
3. 将量化信号与新闻事件交叉验证，指出一致性或矛盾之处
4. 指出潜在风险因素和关键关注点
5. 报告使用中文，包含以下板块：

## 市场情绪判断
（基于新闻+数据综合，判断当前市场情绪偏多/偏空/中性）

## 信号与新闻交叉验证
（量化信号方向是否得到新闻/宏观事件支撑？有无背离？）

## 关键风险提示
（列出当前最值得关注的3-5个风险因素）

## 短期展望
（1-3个交易日的操作建议和关注要点）"""


class LLMAnalyzer:
    """
    Orchestrate quantitative data + news fetching + DeepSeek API analysis.
    """

    def __init__(self, ctx, cache_dir=None):
        """
        Parameters:
            ctx: StrategyContext instance (from strategy_agent.py)
            cache_dir: news cache directory (defaults to <base>/data/macro)
        """
        self.ctx = ctx
        if cache_dir is None:
            cache_dir = os.path.join(ctx.base_dir, "data", "macro")
        self.cache_dir = cache_dir
        self.news_fetcher = NewsDataFetcher(cache_dir=cache_dir)

        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

    def _build_quantitative_context(self):
        """Assemble a compact Chinese-language quantitative snapshot for the LLM prompt."""
        lines = []

        # --- Signal ---
        signal = self.ctx.signal
        if signal:
            direction = signal.get('direction', 0)
            dir_name = signal.get('direction_name', '观望')
            confidence = signal.get('confidence', 0)
            weight = signal.get('suggested_weight', 0)
            regime = signal.get('regime_name', '未知')
            pred_raw = signal.get('predicted_return', 0)
            pred_smooth = signal.get('predicted_return_smooth', 0)
            timestamp = signal.get('timestamp', 'N/A')

            lines.append(f"## 当前信号 ({timestamp})")
            lines.append(f"- 方向: {dir_name} (confidence={confidence:.1%}, weight={weight:.1%})")
            lines.append(f"- 市场状态: {regime}")
            lines.append(f"- 预测收益: raw={pred_raw:+.4f}%, smooth={pred_smooth:+.4f}%")

        # --- Factor extremes ---
        imp = self.ctx.importance
        df = self.ctx.df_factors
        if imp is not None and df is not None and len(df) > 0:
            top_features = imp[imp['importance'] > 0].head(15)['feature'].tolist()
            available = [f for f in top_features if f in df.columns]
            latest = df.iloc[-1]

            extremes = []
            for feat in available:
                series = df[feat].dropna()
                if len(series) < 100:
                    continue
                cur = latest[feat]
                pct = (series < cur).mean() * 100
                if pct > 75 or pct < 25:
                    direction = '偏高' if pct > 75 else '偏低'
                    extremes.append(f"  {feat}: {cur:.4f} (历史 {pct:.0f}% 分位, {direction})")

            if extremes:
                lines.append(f"\n## 异常因子 (前{min(10, len(extremes))}个)")
                lines.extend(extremes[:10])

        # --- Macro snapshot ---
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            macro_keys = [
                ('YC_Slope_30Y_10Y', '30Y-10Y期限利差'),
                ('CN_US_10Y_Spread', '中美10Y利差'),
                ('PMI_ZScore', 'PMI Z-Score'),
                ('CPI_Momentum', 'CPI 动量'),
                ('M2_Surprise', 'M2 超预期'),
                ('Macro_Surprise_Composite', '宏观综合意外'),
                ('Risk_On_Off', 'Risk-On/Off'),
            ]
            lines.append("\n## 宏观快照")
            for col, label in macro_keys:
                if col in df.columns:
                    series = df[col].dropna()
                    cur = latest[col]
                    pct = (series < cur).mean() * 100 if len(series) > 0 else 50
                    lines.append(f"  {label}: {cur:+.4f} (分位 {pct:.0f}%)")

        # --- Performance ---
        metrics = self.ctx.metrics
        if metrics:
            lines.append("\n## 回测表现")
            for key in ['夏普比率', '最大回撤', '日胜率', 'Calmar比率']:
                if key in metrics:
                    lines.append(f"  {key}: {metrics[key]}")

        return "\n".join(lines) if len(lines) > 1 else "(暂无足够量化数据)"

    def _fetch_news_context(self):
        """Fetch latest bond market news, format as LLM context."""
        try:
            data = self.news_fetcher.fetch_all()
            df = data.get('bond_news', pd.DataFrame())
            if len(df) == 0:
                return "（今日暂无债券市场相关新闻数据）"

            recent = df.head(10)
            lines = []
            for i, row in recent.iterrows():
                title = row.get('title', '无标题')
                src = row.get('source', '未知来源')
                dt = str(row.get('date', ''))[:10]
                lines.append(f"{i+1}. [{dt}] {title} (来源: {src})")

            return "\n".join(lines)
        except Exception as e:
            print(f"[LLMAnalyzer] News fetch failed: {e}")
            return "（新闻数据获取失败）"

    def generate_intelligence_report(self, save=True):
        """
        Main entry point: assemble context, call DeepSeek, return report.
        """
        if not self.api_key:
            return self._fallback_report("未设置 DEEPSEEK_API_KEY 环境变量")

        # 1. Build context
        print("[LLMAnalyzer] 构建量化数据上下文...")
        quant_context = self._build_quantitative_context()

        print("[LLMAnalyzer] 获取市场新闻...")
        news_context = self._fetch_news_context()

        # 2. Build prompts
        user_prompt = f"""## 当前量化数据快照

{quant_context}

## 最新债券市场新闻

{news_context}

请基于以上数据，生成今日TL国债期货市场情报分析报告。"""

        # 3. Call DeepSeek
        print(f"[LLMAnalyzer] 调用 DeepSeek ({self.model})...")
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
                temperature=0.3,
            )
            report = response.choices[0].message.content
            print(f"[LLMAnalyzer] DeepSeek 响应: {len(report)} 字符")

        except Exception as e:
            print(f"[LLMAnalyzer] DeepSeek API 调用失败: {e}")
            return self._fallback_report(f"API 调用失败: {e}")

        # 4. Format output
        full_report = "\n".join([
            "=" * 56,
            "  TL国债期货 — AI 市场情报分析",
            "=" * 56,
            f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  模型: {self.model}",
            f"  数据截止: {self._data_timestamp()}",
            "=" * 56,
            "",
            report,
            "",
            "=" * 56,
            "  免责声明: AI生成内容仅供参考，不构成投资建议",
            "=" * 56,
        ])

        # 5. Save
        if save:
            self._save_report(full_report, quant_context, news_context)

        return full_report

    def _save_report(self, report_text, quant_context, news_context):
        """Save report to outputs/ai_intelligence.json."""
        output_dir = os.path.join(self.ctx.base_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "ai_intelligence.json")

        record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model": self.model,
            "report": report_text,
            "data_snapshot": {
                "signal_direction": self.ctx.signal.get('direction', 0) if self.ctx.signal else None,
                "signal_confidence": self.ctx.signal.get('confidence', 0) if self.ctx.signal else None,
                "regime": self.ctx.signal.get('regime_name', 'N/A') if self.ctx.signal else 'N/A',
            }
        }

        # Append to history file
        history_path = os.path.join(output_dir, "ai_intelligence_history.json")
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        # Keep last 50 records
        history = history[-50:]

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # Also save latest as standalone
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"[LLMAnalyzer] 报告已保存: {output_path}")

    def _data_timestamp(self):
        """Get the latest data timestamp for display."""
        df = self.ctx.df_factors
        if df is not None and len(df) > 0 and 'date' in df.columns:
            return str(df['date'].iloc[-1])
        return 'N/A'

    def _fallback_report(self, reason):
        """Generate rule-based report when LLM is unavailable."""
        lines = []
        lines.append("=" * 56)
        lines.append("  TL国债期货 — 市场情报分析 (离线模式)")
        lines.append("=" * 56)
        lines.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  [注意] AI 分析不可用: {reason}")
        lines.append("  以下为基于规则的量化摘要：")
        lines.append("=" * 56)

        # Append rule-based summaries
        lines.append("")
        lines.append("## 当前信号")
        signal = self.ctx.signal
        if signal:
            d = signal.get('direction', 0)
            dir_name = {1: '做多 LONG', -1: '做空 SHORT', 0: '观望 FLAT'}[d]
            lines.append(f"  方向: {dir_name}")
            lines.append(f"  置信度: {signal.get('confidence', 0):.1%}")
            lines.append(f"  市场状态: {signal.get('regime_name', 'N/A')}")
            lines.append(f"  建议仓位: {signal.get('suggested_weight', 0):.1%}")
        else:
            lines.append("  (无信号数据)")

        # Factor diagnostics summary
        imp = self.ctx.importance
        df = self.ctx.df_factors
        if imp is not None and df is not None and len(df) > 0:
            lines.append("")
            lines.append("## 因子诊断")
            top_features = imp[imp['importance'] > 0].head(10)['feature'].tolist()
            available = [f for f in top_features if f in df.columns]
            latest = df.iloc[-1]
            alerts = []
            for feat in available:
                series = df[feat].dropna()
                if len(series) < 100:
                    continue
                cur = latest[feat]
                pct = (series < cur).mean() * 100
                if pct > 75 or pct < 25:
                    direction = '偏高' if pct > 75 else '偏低'
                    alerts.append(f"  !! {feat}: {cur:.4f} (分位 {pct:.0f}%, {direction})")

            if alerts:
                lines.extend(alerts[:8])
            else:
                lines.append("  核心因子均在正常范围内")

        lines.append("")
        lines.append("=" * 56)
        lines.append("  设置 DEEPSEEK_API_KEY 环境变量以启用 AI 分析")
        lines.append("=" * 56)

        return "\n".join(lines)


# ============================================================
# Standalone test
# ============================================================
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from strategy_agent import StrategyContext

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ctx = StrategyContext(base_dir)

    llm = LLMAnalyzer(ctx)
    print(llm.generate_intelligence_report(save=True))
