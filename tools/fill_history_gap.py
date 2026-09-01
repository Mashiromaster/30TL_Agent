# -*- coding: utf-8 -*-
# fill_history_gap.py — 用多周期数据填补原始分钟数据缺口
# Sina API 每次返回 ~1023 行, period='15' 可回到3月, period='30' 可回到12月

import pandas as pd
import numpy as np
import os
import time
import argparse
import akshare as ak

DEFAULT_BASE_DIR = r"D:\桌面\F_Agent"

VARIETY_CONFIG = {
    'TL': {
        'label': 'TL (30年期)',
        'prefix': 'TL',
        'file_name': 'TL分钟级量价数据.pkl',
        'main_symbol': 'TL0',
        'contracts': ['TL2609', 'TL2606', 'TL2603', 'TL2512', 'TL2509'],
        'oi_thresholds': [(100000, 'TL2609'), (50000, 'TL2606'), (20000, 'TL2603')],
        'oi_fallback': 'TL2512',
    },
    'T': {
        'label': 'T (10年期)',
        'prefix': 'T',
        'file_name': 'T分钟级量价数据.pkl',
        'main_symbol': 'T0',
        'contracts': ['T2609', 'T2606', 'T2603', 'T2512', 'T2509'],
        'oi_thresholds': [(80000, 'T2609'), (40000, 'T2606'), (15000, 'T2603')],
        'oi_fallback': 'T2512',
    },
    'TF': {
        'label': 'TF (5年期)',
        'prefix': 'TF',
        'file_name': 'TF分钟级量价数据.pkl',
        'main_symbol': 'TF0',
        'contracts': ['TF2609', 'TF2606', 'TF2603', 'TF2512', 'TF2509'],
        'oi_thresholds': [(50000, 'TF2609'), (25000, 'TF2606'), (10000, 'TF2603')],
        'oi_fallback': 'TF2512',
    },
    'TS': {
        'label': 'TS (2年期)',
        'prefix': 'TS',
        'file_name': 'TS分钟级量价数据.pkl',
        'main_symbol': 'TS0',
        'contracts': ['TS2609', 'TS2606', 'TS2603', 'TS2512', 'TS2509'],
        'oi_thresholds': [(30000, 'TS2609'), (15000, 'TS2606'), (5000, 'TS2603')],
        'oi_fallback': 'TS2512',
    },
}


def _infer_ticker(sym, df, variety='TL'):
    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG['TL'])
    if sym != cfg['main_symbol']:
        return sym
    oi_last = float(df['hold'].iloc[-1])
    for threshold, contract in cfg['oi_thresholds']:
        if oi_last > threshold:
            return contract
    return cfg['oi_fallback']


def fetch_period(variety, period):
    """Fetch data for given period ('1', '15', '30') for the given variety."""
    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG['TL'])
    symbols = [cfg['main_symbol']] + cfg['contracts']
    main_sym = cfg['main_symbol']

    frames = []
    for sym in symbols:
        try:
            raw = ak.futures_zh_minute_sina(symbol=sym, period=period)
            if raw is None or len(raw) == 0:
                print(f"  {sym}: 无数据")
                continue

            ticker = _infer_ticker(sym, raw, variety)
            df_out = pd.DataFrame()
            df_out['date'] = pd.to_datetime(raw['datetime'])
            df_out['trade_dt'] = df_out['date'].dt.normalize()
            df_out['ticker'] = ticker
            df_out['open'] = raw['open'].astype(float)
            df_out['high'] = raw['high'].astype(float)
            df_out['low'] = raw['low'].astype(float)
            df_out['close'] = raw['close'].astype(float)
            df_out['volume'] = raw['volume'].astype(float)
            df_out['money'] = df_out['close'] * df_out['volume'] * 100
            df_out['oi'] = raw['hold'].astype(float)
            df_out['time'] = df_out['date'].dt.strftime('%H:%M')

            frames.append(df_out)
            print(f"  {sym} → {ticker}: {len(df_out)} 行, "
                  f"{df_out['date'].min()} ~ {df_out['date'].max()}")
        except Exception as e:
            print(f"  {sym}: 失败 — {e}")
        time.sleep(0.3)
    return frames


def main(base_dir=DEFAULT_BASE_DIR, variety='TL'):
    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG['TL'])
    raw_file = os.path.join(base_dir, "data", cfg['file_name'])

    print("=" * 55)
    print(f"  填补 {cfg['label']} 分钟数据缺口")
    print("=" * 55)

    # 1. 加载现有数据
    if os.path.exists(raw_file):
        df_existing = pd.read_pickle(raw_file)
        print(f"\n现有数据: {len(df_existing):,} 行, "
              f"{df_existing['date'].min()} ~ {df_existing['date'].max()}")
    else:
        df_existing = pd.DataFrame()

    all_new = []

    # 2. 按周期从细到粗拉取
    for period in ['1', '15', '30']:
        label = {'1': '1分钟', '15': '15分钟', '30': '30分钟'}[period]
        print(f"\n--- 拉取 {label} 数据 ---")
        frames = fetch_period(variety, period)
        if frames:
            df_period = pd.concat(frames, ignore_index=True)
            print(f"  合计: {len(df_period)} 行")
            all_new.append(df_period)

    if not all_new:
        print("\n未获取到任何数据")
        return

    df_new = pd.concat(all_new, ignore_index=True)

    # 3. 合并去重 (同 date+ticker 保留先出现的=细粒度优先)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['date', 'ticker'], keep='first')
    after = len(df_combined)
    added = after - len(df_existing)

    df_combined = df_combined.sort_values(['date', 'ticker']).reset_index(drop=True)

    # 4. 保存
    df_combined.to_pickle(raw_file)
    print(f"\n{'=' * 55}")
    print(f"  保存完成: {raw_file}")
    print(f"  去重: {before:,} → {after:,} (新增 {added:,} 行)")
    print(f"  日期范围: {df_combined['date'].min()} ~ {df_combined['date'].max()}")
    print(f"  合约: {sorted(df_combined['ticker'].unique())}")

    # 月度覆盖统计
    df_combined['trade_day'] = df_combined['date'].dt.normalize()
    df_combined['month'] = df_combined['trade_day'].dt.to_period('M')
    monthly = df_combined.groupby('month')['trade_day'].nunique()
    print(f"\n  月度交易日覆盖:")
    for m, cnt in monthly.tail(12).items():
        print(f"    {m}: {cnt} 天")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fill historical minute data gaps')
    parser.add_argument('--variety', type=str, default='TL',
                        choices=['TL', 'T', 'TF', 'TS'],
                        help='Futures variety to fill (default: TL)')
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR,
                        help='Project root directory')
    args = parser.parse_args()
    main(args.base_dir, args.variety)
