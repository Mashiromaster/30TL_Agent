# -*- coding: utf-8 -*-
# update_market_data.py — 从 AKShare 拉取 TL 期货分钟数据并追加到原始数据

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

RAW_FILE = r"D:\桌面\F_Agent\data\TL分钟级量价数据.pkl"


def fetch_tl_contracts():
    """
    拉取所有可获取的 TL 合约分钟数据。
    返回 list of DataFrame，每个 DataFrame 的列与原始数据一致。
    """
    import akshare as ak

    # 当前和最近的主力合约
    symbols = ['TL0', 'TL2609', 'TL2606', 'TL2603', 'TL2512', 'TL2509']

    results = []
    for sym in symbols:
        try:
            df = ak.futures_zh_minute_sina(symbol=sym, period='1')
            if df is None or len(df) == 0:
                print(f"  {sym}: 无数据")
                continue

            # Determine actual ticker: TL0 resolves to current main contract
            # We infer the actual contract from hold values
            ticker = sym if sym != 'TL0' else _infer_ticker(sym, df)

            # Transform columns to match existing raw data
            df_out = pd.DataFrame()
            df_out['date'] = pd.to_datetime(df['datetime'])
            df_out['trade_dt'] = df_out['date'].dt.normalize()
            df_out['ticker'] = ticker
            df_out['open'] = df['open'].astype(float)
            df_out['high'] = df['high'].astype(float)
            df_out['low'] = df['low'].astype(float)
            df_out['close'] = df['close'].astype(float)
            df_out['volume'] = df['volume'].astype(float)
            # 成交额 = 价格 * 成交量 * 合约乘数 (TL=10000面额, 价格单位是元/百元)
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


def _infer_ticker(sym, df):
    """TL0 返回的是当前主力合约, 根据 OI 范围推断合约代码."""
    oi_last = df['hold'].iloc[-1]
    if oi_last > 100000:
        return 'TL2609'
    elif oi_last > 50000:
        return 'TL2606'
    elif oi_last > 20000:
        return 'TL2603'
    return 'TL2512'


def update_raw_data():
    """主流程：拉取新数据 → 去重 → 追加 → 保存"""
    print("=" * 50)
    print("  TL 分钟行情数据更新")
    print("=" * 50)

    # 1. 加载现有数据
    if os.path.exists(RAW_FILE):
        df_existing = pd.read_pickle(RAW_FILE)
        print(f"\n现有数据: {len(df_existing):,} 行, "
              f"{df_existing['date'].min()} ~ {df_existing['date'].max()}")
        print(f"合约: {sorted(df_existing['ticker'].unique())}")
    else:
        print(f"\n[WARNING] 未找到 {RAW_FILE}, 将创建新文件")
        df_existing = pd.DataFrame()

    # 2. 拉取新数据
    print(f"\n拉取 TL 合约分钟数据...")
    new_dfs = fetch_tl_contracts()

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
    df_combined.to_pickle(RAW_FILE)
    print(f"\n已保存: {RAW_FILE}")
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
    update_raw_data()
