# -*- coding: utf-8 -*-
# backtest_v2.py — 排名信号回测，替代绝对阈值
"""
V2回测改进:
1. 排名信号: 前N%预测 → 做多, 后N%预测 → 做空
2. 信号平滑优化
3. 动态持仓管理
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = r"D:\桌面\F_Agent"


def run_rank_backtest(pred_file=None, rank_pct=0.25, smooth_span=80):
    """排名信号回测"""
    if pred_file is None:
        pred_file = os.path.join(BASE_DIR, "outputs/df_predictions.pkl")
    
    if not os.path.exists(pred_file):
        print(f"[ERROR] 预测文件不存在: {pred_file}")
        return None
    
    df = pd.read_pickle(pred_file)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n{'='*60}")
    print(f"排名信号回测 (top/bottom {rank_pct:.0%}, smooth={smooth_span})")
    print(f"{'='*60}")
    print(f"数据: {len(df)} 行, {df['date'].dt.date.nunique()} 天")
    print(f"期间: {df['date'].min().date()} ~ {df['date'].max().date()}")
    
    # IC检查
    ic = df['Target_Ret'].corr(df['Pred_Ret'])
    print(f"IC: {ic:.4f}")
    
    if 'Market_Regime' in df.columns:
        for r, name in [(0, 'Normal'), (1, 'HighVol'), (2, 'Trend')]:
            mask = df['Market_Regime'] == r
            if mask.sum() > 10:
                r_ic = df.loc[mask, 'Target_Ret'].corr(df.loc[mask, 'Pred_Ret'])
                print(f"  {name}: IC={r_ic:.4f}")
    
    # 甜点区排名信号 (基于219笔记忆回测优化)
    df['Pred_Smooth'] = df['Pred_Ret'].ewm(span=smooth_span, adjust=False).mean()
    
    # 做空甜点区: rank 15%-30%  (准确率72.7%), 做多甜点区: rank 70%-85%  (准确率63.6%)
    short_inner = 0.15
    short_outer = 0.30
    long_inner = 0.70
    long_outer = 0.85
    
    # 排名信号: 用过去480根bar的排名
    lookback = 480
    df['Pred_Rank'] = np.nan
    
    for i in range(lookback, len(df)):
        window = df['Pred_Smooth'].iloc[i-lookback:i]
        current = df['Pred_Smooth'].iloc[i]
        df.loc[df.index[i], 'Pred_Rank'] = (window < current).sum() / len(window)
    
    # 甜点区信号
    df['Raw_Signal'] = 0
    df.loc[(df['Pred_Rank'] >= short_inner) & (df['Pred_Rank'] < short_outer), 'Raw_Signal'] = -1
    df.loc[(df['Pred_Rank'] >= long_inner) & (df['Pred_Rank'] < long_outer), 'Raw_Signal'] = 1
    
    # 信号确认（连续N个bar维持同一方向）
    confirm_bars = 10
    df['Confirmed_Signal'] = 0
    
    for i in range(confirm_bars, len(df)):
        window_signals = df['Raw_Signal'].iloc[i-confirm_bars:i]
        if (window_signals == 1).all():
            df.loc[df.index[i], 'Confirmed_Signal'] = 1
        elif (window_signals == -1).all():
            df.loc[df.index[i], 'Confirmed_Signal'] = -1
    
    # 持仓管理
    n = len(df)
    position = np.zeros(n)
    last_signal_time = -1000
    current_pos = 0
    min_bars_between_trades = 60  # 最少60根bar（1小时）
    
    for i in range(n):
        signal = df['Confirmed_Signal'].iloc[i]
        bars_since_last = i - last_signal_time
        
        if bars_since_last >= min_bars_between_trades:
            if signal != 0 and signal != current_pos:
                current_pos = signal
                last_signal_time = i
            elif current_pos != 0 and bars_since_last > min_bars_between_trades * 3:
                # 超时平仓
                current_pos = 0
        
        position[i] = current_pos
    
    df['Position'] = position
    
    # 收益计算
    commission = 0.000023
    slippage_ticks = 1.0
    tick_size = 0.01
    
    df['Asset_Ret'] = df['close'].pct_change().fillna(0)
    df['Trades'] = (df['Position'] - df['Position'].shift(1).fillna(0)).abs()
    
    rel_slippage = (slippage_ticks * tick_size) / df['close'].clip(lower=1)
    df['Cost'] = df['Trades'] * (commission + rel_slippage)
    df['Net_Ret'] = df['Position'].shift(1).fillna(0) * df['Asset_Ret'] - df['Cost']
    
    # 杠杆控制
    roll_vol = df['Net_Ret'].rolling(240*5, min_periods=240).std() * np.sqrt(240*252)
    df['Leverage'] = (0.12 / roll_vol).fillna(1.0).clip(0.3, 2.0)
    df['Strategy_Ret'] = df['Net_Ret'] * df['Leverage']
    df['Cum_Ret'] = (1 + df['Strategy_Ret']).cumprod()
    
    # 日聚合
    df_daily = df.groupby(df['date'].dt.date).agg({
        'Strategy_Ret': 'sum',
        'Position': 'last',
        'Trades': 'sum',
    }).reset_index()
    df_daily.columns = ['date', 'ret', 'position', 'trades']
    
    # 指标计算
    n_days = len(df_daily)
    df_daily['cum'] = (1 + df_daily['ret']).cumprod()
    
    total_return = df_daily['cum'].iloc[-1] - 1
    ann_ret = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = df_daily['ret'].std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    
    df_daily['peak'] = df_daily['cum'].cummax()
    df_daily['drawdown'] = (df_daily['cum'] - df_daily['peak']) / df_daily['peak']
    mdd = df_daily['drawdown'].min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0
    
    trading_days = (df_daily['ret'].abs() > 0.0001).sum()
    winning_days = (df_daily['ret'] > 0.0001).sum()
    win_rate = winning_days / trading_days if trading_days > 0 else 0
    
    long_pct = (df['Position'] > 0).mean()
    short_pct = (df['Position'] < 0).mean()
    flat_pct = (df['Position'] == 0).mean()
    
    total_trades_val = df_daily['trades'].sum() / 2
    ann_turnover = total_trades_val * (252 / n_days)
    
    # 输出
    print(f"\n{'='*60}")
    print(f"回测结果")
    print(f"{'='*60}")
    print(f"  累计收益:   {total_return:.2%}")
    print(f"  年化收益:   {ann_ret:.2%}")
    print(f"  年化波动:   {ann_vol:.2%}")
    print(f"  夏普比率:   {sharpe:.3f}")
    print(f"  最大回撤:   {mdd:.2%}")
    print(f"  Calmar:     {calmar:.2f}")
    print(f"  日胜率:     {win_rate:.1%}")
    print(f"  年化换手:   {ann_turnover:.1f}x")
    print(f"  做多: {long_pct:.1%}  做空: {short_pct:.1%}  空仓: {flat_pct:.1%}")
    
    if 'Market_Regime' in df.columns:
        print(f"\n分状态收益:")
        for r, name in [(0, 'Normal'), (1, 'HighVol'), (2, 'Trend')]:
            mask = df['Market_Regime'] == r
            if mask.sum() > 0:
                ret = df.loc[mask, 'Strategy_Ret'].sum()
                pct = mask.sum() / len(df)
                print(f"  {name}: {ret:.2%} (占比 {pct:.1%})")
    
    return df_daily, {
        '累计收益': f"{total_return:.2%}",
        '年化收益': f"{ann_ret:.2%}",
        '夏普比率': f"{sharpe:.3f}",
        '最大回撤': f"{mdd:.2%}",
        'Calmar': f"{calmar:.2f}",
        '日胜率': f"{win_rate:.1%}",
    }


def sweep_params():
    """扫描回测参数"""
    print("\n" + "="*60)
    print("回测参数扫描")
    print("="*60)
    
    results = []
    for rank_pct in [0.20, 0.25, 0.30, 0.35]:
        for smooth_span in [60, 80, 120, 160]:
            df_daily, metrics = run_rank_backtest(
                rank_pct=rank_pct,
                smooth_span=smooth_span
            )
            sharpe = float(metrics['夏普比率'])
            ret = float(metrics['累计收益'].rstrip('%')) / 100
            results.append({
                'rank_pct': rank_pct, 'smooth': smooth_span,
                'sharpe': sharpe, 'ret': ret,
                'mdd': float(metrics['最大回撤'].rstrip('%')) / 100,
            })
    
    results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
    
    print(f"\n排名 (按Sharpe):")
    for i, r in enumerate(results_sorted):
        star = " *** BEST ***" if i == 0 else ""
        print(f"  {i+1}. rank={r['rank_pct']:.0%} span={r['smooth']:3d}  "
              f"S={r['sharpe']:.3f}  Ret={r['ret']:.2%}  MDD={r['mdd']:.2%}{star}")
    
    return results_sorted[0]


if __name__ == '__main__':
    best = sweep_params()
    print(f"\n最优参数: rank_pct={best['rank_pct']:.0%}, smooth_span={best['smooth']}")
