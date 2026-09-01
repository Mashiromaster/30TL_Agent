# -*- coding: utf-8 -*-
# update_market_data.py — 从 AKShare 拉取国债期货分钟数据并追加到原始数据

import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime

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
    """Infer actual contract code from OI for the given variety's main-contract symbol."""
    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG['TL'])
    if sym != cfg['main_symbol']:
        return sym
    oi_last = float(df['hold'].iloc[-1])
    for threshold, contract in cfg['oi_thresholds']:
        if oi_last > threshold:
            return contract
    return cfg['oi_fallback']


def fetch_contracts(variety='TL'):
    """拉取指定品种的所有合约分钟数据。"""
    import akshare as ak

    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG['TL'])
    symbols = [cfg['main_symbol']] + cfg['contracts']
    main_sym = cfg['main_symbol']

    results = []
    for sym in symbols:
        try:
            df = ak.futures_zh_minute_sina(symbol=sym, period='1')
            if df is None or len(df) == 0:
                print(f"  {sym}: 无数据")
                continue

            ticker = _infer_ticker(sym, df, variety)

            df_out = pd.DataFrame()
            df_out['date'] = pd.to_datetime(df['datetime'])
            df_out['trade_dt'] = df_out['date'].dt.normalize()
            df_out['ticker'] = ticker
            df_out['open'] = df['open'].astype(float)
            df_out['high'] = df['high'].astype(float)
            df_out['low'] = df['low'].astype(float)
            df_out['close'] = df['close'].astype(float)
            df_out['volume'] = df['volume'].astype(float)
            df_out['money'] = df_out['close'] * df_out['volume'] * 100
            df_out['oi'] = df['hold'].astype(float)
            df_out['time'] = df_out['date'].dt.strftime('%H:%M')

            results.append(df_out)
            print(f"  {sym} → {ticker}: {len(df_out)} 行, "
                  f"{df_out['date'].min()} ~ {df_out['date'].max()}, OI={df_out['oi'].iloc[-1]:.0f}")

        except Exception as e:
            print(f"  {sym}: 失败 — {e}")

        time.sleep(0.5)

    return results


def update_raw_data(base_dir=DEFAULT_BASE_DIR, variety='TL'):
    """主流程：拉取新数据 → 去重 → 追加 → 保存"""
    cfg = VARIETY_CONFIG.get(variety, VARIETY_CONFIG['TL'])
    raw_file = os.path.join(base_dir, "data", cfg['file_name'])

    print("=" * 50)
    print(f"  {cfg['label']} 分钟行情数据更新")
    print("=" * 50)

    # 1. 加载现有数据
    if os.path.exists(raw_file):
        df_existing = pd.read_pickle(raw_file)
        print(f"\n现有数据: {len(df_existing):,} 行, "
              f"{df_existing['date'].min()} ~ {df_existing['date'].max()}")
        print(f"合约: {sorted(df_existing['ticker'].unique())}")
    else:
        print(f"\n[WARNING] 未找到 {raw_file}, 将创建新文件")
        df_existing = pd.DataFrame()

    # 2. 拉取新数据
    print(f"\n拉取 {cfg['label']} 合约分钟数据...")
    new_dfs = fetch_contracts(variety=variety)

    if not new_dfs:
        print("[ERROR] 未获取到任何数据")
        return False

    df_new = pd.concat(new_dfs, ignore_index=True)
    print(f"\n新数据合计: {len(df_new):,} 行")

    # 3. 合并并去重 (按 date + ticker)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['date', 'ticker'], keep='last')
    after = len(df_combined)
    print(f"去重: {before:,} → {after:,} (新增 {after - len(df_existing):,} 行)")

    df_combined = df_combined.sort_values(['date', 'ticker']).reset_index(drop=True)

    # 4. 保存
    df_combined.to_pickle(raw_file)
    print(f"\n已保存: {raw_file}")
    print(f"总行数: {len(df_combined):,}")
    print(f"日期范围: {df_combined['date'].min()} ~ {df_combined['date'].max()}")
    print(f"合约: {sorted(df_combined['ticker'].unique())}")

    # 5. 覆盖摘要
    last = df_combined['date'].max()
    days_to_now = (datetime.now() - pd.to_datetime(last)).days
    print(f"最新数据距今天: {days_to_now} 天")
    if days_to_now > 1:
        print("[WARNING] 数据有滞后, 请定期运行此脚本更新")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update treasury futures minute data')
    parser.add_argument('--variety', type=str, default='TL',
                        choices=['TL', 'T', 'TF', 'TS'],
                        help='Futures variety to update (default: TL)')
    parser.add_argument('--base-dir', type=str, default=DEFAULT_BASE_DIR,
                        help='Project root directory')
    args = parser.parse_args()
    update_raw_data(args.base_dir, args.variety)
