# -*- coding: utf-8 -*-
# text_factor.py — 研报文本情绪因子 (Policy Sentiment Factor, V1.0)
#
# spec: docs/superpowers/specs/2026-09-01-policy-sentiment-factor-design.md
#
# 把宏观研报(非结构化文本)转成日频情绪因子,前向填充到分钟 bar 进模型训练。
# 关键: 严格防前视偏差 —— 只用有真实发布日的研报, available_date = 发布日 + 1天。

import os
import glob

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

    返回 DataFrame[pub_date(datetime.date), text] 或空 DataFrame。
    """
    rag_dir = os.path.join(base_dir, "data", "rag")
    if not os.path.isdir(rag_dir):
        return pd.DataFrame(columns=['pub_date', 'text'])

    rows = []
    for pkl in glob.glob(os.path.join(rag_dir, "*.pkl")):
        try:
            df = pd.read_pickle(pkl)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or 'date' not in df.columns:
            continue  # 无真实发布日字段 → 整源丢弃(防前视)
        for _, r in df.iterrows():
            pub = pd.to_datetime(r.get('date'), errors='coerce')
            if pd.isna(pub):
                continue  # 发布日不可解析 → 丢弃
            text = str(r.get('content', '') or r.get('title', ''))
            rows.append({'pub_date': pub.date(), 'text': text})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['pub_date', 'text'])


def build_sentiment_factor(base_dir):
    """构建日频文本情绪因子。

    返回 DataFrame[available_date, Policy_Sentiment, Policy_Sentiment_MA5]。
    无可用研报时返回空 DataFrame (调用方降级为全 0)。

    防前视: available_date = 发布日 + 1 天 (T+1 生效)。
    """
    reports = _load_reports_with_dates(base_dir)
    if len(reports) == 0:
        print("[TextFactor] 无带真实发布日的研报 → 情绪因子将为全 0")
        return pd.DataFrame(columns=['available_date', 'Policy_Sentiment', 'Policy_Sentiment_MA5'])

    reports['score'] = reports['text'].apply(score_text)

    # 日频聚合: 同一发布日多份研报取均值
    daily = reports.groupby('pub_date', as_index=False)['score'].mean()
    daily = daily.rename(columns={'score': 'Policy_Sentiment'})
    daily = daily.sort_values('pub_date').reset_index(drop=True)

    # 5 日均值降噪 (按发布日序列)
    daily['Policy_Sentiment_MA5'] = daily['Policy_Sentiment'].rolling(5, min_periods=1).mean()

    # 防前视: T+1 生效
    daily['available_date'] = pd.to_datetime(daily['pub_date']) + pd.Timedelta(days=1)

    out = daily[['available_date', 'Policy_Sentiment', 'Policy_Sentiment_MA5']].copy()
    print(f"[TextFactor] 情绪因子: {len(out)} 个发布日, "
          f"均值 {out['Policy_Sentiment'].mean():+.3f}, "
          f"范围 [{out['Policy_Sentiment'].min():+.3f}, {out['Policy_Sentiment'].max():+.3f}]")
    return out
