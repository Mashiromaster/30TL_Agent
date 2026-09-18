# -*- coding: utf-8 -*-
# text_factor.py — 研报文本情绪因子 (Policy Sentiment Factor, V1.0)
#
# spec: docs/superpowers/specs/2026-09-01-policy-sentiment-factor-design.md
#
# 把宏观研报转成事件频率情绪因子，并按真实可用时点合并。

import os
from pathlib import Path

import numpy as np
import pandas as pd

# 关键词词典 (与 signal_fusion.py 一致)
BULL_KEYWORDS = ['降准', '降息', '宽松', '利多', '下行', '回落', '流动性充裕', '放水', '再贷款']
BEAR_KEYWORDS = ['加息', '紧缩', '利空', '上行', '通胀', '收紧', '流动性紧张', '去杠杆']


def score_text(text):
    """对单条文本按关键词词频打分 → [-1, 1]。无命中返回 0。"""
    if not text:
        return 0.0
    bull = sum(text.count(kw) for kw in BULL_KEYWORDS)
    bear = sum(text.count(kw) for kw in BEAR_KEYWORDS)
    if bull + bear == 0:
        return 0.0
    return (bull - bear) / (bull + bear + 1)


def _load_reports_with_dates(base_dir):
    """从 data/rag 缓存读研报, 只保留有可解析真实发布日的行。

    返回 DataFrame[pub_ts, has_intraday_ts, text] 或空 DataFrame。
    """
    rag_dir = os.path.join(base_dir, "data", "rag")
    if not os.path.isdir(rag_dir):
        return pd.DataFrame(columns=['pub_ts', 'has_intraday_ts', 'text'])

    rows = []
    for pkl in Path(rag_dir).rglob("*.pkl"):
        try:
            df = pd.read_pickle(pkl)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or 'date' not in df.columns:
            continue  # 无真实发布日字段 → 整源丢弃(防前视)
        for _, r in df.iterrows():
            raw_date = r.get('date')
            pub = pd.to_datetime(raw_date, errors='coerce')
            if pd.isna(pub):
                continue  # 发布日不可解析 → 丢弃
            text = str(r.get('content', '') or r.get('title', ''))
            has_intraday = (
                pub.hour != 0 or pub.minute != 0 or pub.second != 0
            )
            rows.append({
                'pub_ts': pub,
                'has_intraday_ts': has_intraday,
                'text': text,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['pub_ts', 'has_intraday_ts', 'text']
    )


def build_sentiment_factor(base_dir):
    """构建日频文本情绪因子。

    返回 DataFrame[available_ts, Policy_Sentiment, Policy_Sentiment_MA5]。
    无可用研报时返回空 DataFrame；不会伪造全零因子。

    有日内发布时间时按该时间生效；只有日期时保守地在下一工作日
    09:00 生效。
    """
    reports = _load_reports_with_dates(base_dir)
    if len(reports) == 0:
        print("[TextFactor] 无带真实发布日的研报 → 不生成文本因子")
        return pd.DataFrame(columns=['available_ts', 'Policy_Sentiment', 'Policy_Sentiment_MA5'])

    reports['score'] = reports['text'].apply(score_text)

    reports['available_ts'] = reports['pub_ts']
    date_only = ~reports['has_intraday_ts']
    reports.loc[date_only, 'available_ts'] = (
        reports.loc[date_only, 'pub_ts'].dt.normalize()
        + pd.offsets.BDay(1)
        + pd.Timedelta(hours=9)
    )

    # 同一可用时点多份研报取均值。
    daily = reports.groupby('available_ts', as_index=False)['score'].mean()
    daily = daily.rename(columns={'score': 'Policy_Sentiment'})
    daily = daily.sort_values('available_ts').reset_index(drop=True)

    # 5 日均值降噪 (按发布日序列)
    daily['Policy_Sentiment_MA5'] = daily['Policy_Sentiment'].rolling(5, min_periods=1).mean()

    out = daily[['available_ts', 'Policy_Sentiment', 'Policy_Sentiment_MA5']].copy()
    out.attrs['factor_ttls'] = {
        'Policy_Sentiment': 10,
        'Policy_Sentiment_MA5': 10,
    }
    print(f"[TextFactor] 情绪因子: {len(out)} 个发布日, "
          f"均值 {out['Policy_Sentiment'].mean():+.3f}, "
          f"范围 [{out['Policy_Sentiment'].min():+.3f}, {out['Policy_Sentiment'].max():+.3f}]")
    return out
