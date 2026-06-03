# -*- coding: utf-8 -*-
# enhanced_factors.py — 新增因子模块：2025-2026市场制度适应
# 涵盖：政策冲击、流动性压力、曲线动态、波动率聚类、机构行为

import pandas as pd
import numpy as np

def add_enhanced_factors(df):
    """
    在已有核心因子的基础上，添加新的制度自适应因子。
    所有因子使用滞后数据，避免未来信息泄露。
    
    新增因子类别:
    1. 政策冲击因子 (Policy Shock) — 无法实时获取，用收益率跳变代理
    2. 流动性压力因子 (Liquidity Stress) — 资金面紧张度
    3. 曲线动态增强 (Curve Dynamics) — 30Y-10Y加速/蝶式
    4. 波动率聚类 (Vol Clustering) — 波动率的结构变化
    5. 机构行为 (Institutional Flow) — OI集中度/持仓变化速度
    6. 趋势强度 (Trend Strength) — ADX/方向运动
    7. 反转风险 (Reversal Risk) — 过度延伸后的回撤概率
    """
    print("[EnhancedFactor] 计算新增制度自适应因子...")
    df = df.copy()
    
    close_lag = df['close'].shift(1)
    ret_lag = close_lag.pct_change()
    
    # ============================================================
    # 1. 政策冲击因子 (用价格行为代理，因为实时政策数据不可得)
    # ============================================================
    # 利率跳变：单日剧烈波动通常反映政策/事件冲击
    daily_ret = ret_lag.copy()
    daily_ret_abs = daily_ret.abs()
    
    # 跳变检测：超过3倍标准差
    ret_std_60 = daily_ret_abs.rolling(60, min_periods=30).std()
    ret_mean_60 = daily_ret.rolling(60, min_periods=30).mean()
    df['Shock_Event'] = (daily_ret_abs > ret_std_60 * 2.5).astype(int)
    
    # 跳变后的均值回复倾向
    df['Shock_Decay_5'] = df['Shock_Event'].rolling(5).sum()
    df['Shock_Decay_20'] = df['Shock_Event'].rolling(20).sum()
    
    # 跳变方向累积
    shock_sign = np.sign(daily_ret) * df['Shock_Event']
    df['Cum_Shock_Sign_10'] = shock_sign.rolling(10).sum()
    
    # ============================================================
    # 2. 流动性压力因子
    # ============================================================
    # 买卖价差压力：用微观结构因子计算
    if 'Spread_Mean' in df.columns:
        spread_lag = df['Spread_Mean'].shift(1)
        spread_ma60 = spread_lag.rolling(60, min_periods=10).mean()
        spread_std60 = spread_lag.rolling(60, min_periods=10).std()
        df['Spread_Stress'] = (spread_lag - spread_ma60) / (spread_std60 + 0.01)
        df['Spread_Stress'] = df['Spread_Stress'].clip(-4, 4)
    
    if 'Depth_Imbalance_Mean' in df.columns:
        depth_lag = df['Depth_Imbalance_Mean'].shift(1)
        depth_ma60 = depth_lag.rolling(60, min_periods=10).mean()
        depth_std60 = depth_lag.rolling(60, min_periods=10).std()
        df['Depth_Stress'] = (depth_lag - depth_ma60) / (depth_std60 + 0.01)
        df['Depth_Stress'] = df['Depth_Stress'].clip(-4, 4)
    
    # ============================================================
    # 3. 曲线动态增强
    # ============================================================
    if 'YC_Slope_30Y_10Y' in df.columns:
        slope_lag = df['YC_Slope_30Y_10Y'].shift(1)
        
        # 曲线斜率变化速度
        df['YC_Slope_Accel'] = slope_lag.diff(5)
        df['YC_Slope_Accel_Z'] = (
            df['YC_Slope_Accel'].rolling(60, min_periods=10).apply(
                lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 0.01) if len(x) >= 10 else 0
            )
        )
        
        # 曲线形态分类：平坦化/陡峭化
        slope_ma20 = slope_lag.rolling(20, min_periods=5).mean()
        slope_ma60 = slope_lag.rolling(60, min_periods=10).mean()
        df['YC_Flattening'] = np.sign(slope_ma20 - slope_ma60)
        
        # 30Y收益率相对于10Y的超额变动
        if 'YC_Level_Shift' in df.columns:
            df['YC_30Y_Excess'] = df['YC_Slope_30Y_10Y'].shift(1).diff(5)
    
    if 'YC_Curvature' in df.columns:
        curve_lag = df['YC_Curvature'].shift(1)
        # 蝶式价差变化
        df['Butterfly_Change_5D'] = curve_lag.diff(5)
    
    # ============================================================
    # 4. 波动率聚类 (Volatility Clustering)
    # ============================================================
    if 'RV_30' in df.columns:
        rv_lag = df['RV_30'].shift(1)
        
        # 波动率的变化率
        df['Vol_of_Vol'] = rv_lag.pct_change(20).clip(-5, 5)
        
        # 波动率持续性（自相关）
        rv_ma5 = rv_lag.rolling(5).mean()
        rv_ma20 = rv_lag.rolling(20).mean()
        df['Vol_Persistence'] = (rv_ma5 / (rv_ma20 + 0.01) - 1).clip(-2, 2)
        
        # 波动率突破
        rv_ma60 = rv_lag.rolling(60, min_periods=20).mean()
        rv_std60 = rv_lag.rolling(60, min_periods=20).std()
        df['Vol_Breakout'] = (rv_lag - rv_ma60) / (rv_std60 + 0.01)
        df['Vol_Breakout'] = df['Vol_Breakout'].clip(-4, 4)
        
        # 高低波动率状态切换信号
        rv_rank = rv_lag.rolling(240, min_periods=60).rank(pct=True)
        df['Vol_Regime_Change'] = (rv_rank - rv_rank.shift(60)).clip(-1, 1)
    
    # ============================================================
    # 5. 机构行为因子
    # ============================================================
    oi_lag = df['oi'].shift(1)
    volume_lag = df['volume'].shift(1)
    
    # OI增速
    df['OI_Growth_5D'] = oi_lag.pct_change(5 * 240).clip(-0.2, 0.2)
    df['OI_Growth_20D'] = oi_lag.pct_change(20 * 240).clip(-0.3, 0.3)
    
    # OI加速度
    df['OI_Acceleration'] = df['OI_Growth_5D'] - df['OI_Growth_5D'].shift(5 * 240)
    
    # 量价配合：放量+持仓变化
    vol_ma20 = volume_lag.rolling(20 * 240, min_periods=60).mean()
    df['Volume_Surge'] = volume_lag / (vol_ma20 + 1)
    df['Volume_Surge'] = df['Volume_Surge'].clip(0.1, 5)
    
    # OI-价格背离
    price_5d = close_lag.pct_change(5 * 240)
    df['OI_Price_Divergence'] = np.sign(price_5d) * np.sign(df['OI_Growth_5D'])
    df['OI_Price_Divergence'] = df['OI_Price_Divergence'].rolling(5 * 240, min_periods=60).mean()
    
    # ============================================================
    # 6. 趋势强度 (ADX-like)
    # ============================================================
    # 方向运动
    high_lag = df['high'].shift(1)
    low_lag = df['low'].shift(1)
    high_prev = df['high'].shift(2)
    low_prev = df['low'].shift(2)
    
    up_move = high_lag - high_prev
    down_move = low_prev - low_lag
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # TR (已在核心因子中)
    tr_series = df['TR'] if 'TR' in df.columns else (high_lag - low_lag)
    tr_smooth = tr_series.rolling(14).mean()
    
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (tr_smooth + 0.01)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (tr_smooth + 0.01)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.01)
    df['ADX'] = dx.rolling(14).mean().clip(0, 100)
    
    # 趋势方向
    df['Trend_Direction'] = np.sign(plus_di - minus_di)
    
    # 趋势强度变化
    df['ADX_Change'] = df['ADX'].diff(5)
    
    # ============================================================
    # 7. 反转风险
    # ============================================================
    # RSI极端值
    if 'RSI' in df.columns:
        rsi_lag = df['RSI'].shift(1)
        df['RSI_Extreme'] = ((rsi_lag > 70) | (rsi_lag < 30)).astype(int)
        df['RSI_Distance_50'] = abs(rsi_lag - 50) / 50
    
    # 连续涨跌天数
    price_sign = np.sign(close_lag.diff())
    df['Consecutive_Up'] = price_sign.replace(-1, 0).rolling(10).apply(
        lambda x: _count_consecutive(x, 1), raw=False
    )
    df['Consecutive_Down'] = price_sign.replace(1, 0).replace(-1, 1).rolling(10).apply(
        lambda x: _count_consecutive(x, 1), raw=False
    )
    
    # BB位置极端
    if 'BB_Position' in df.columns:
        bb_lag = df['BB_Position'].shift(1)
        df['BB_Extreme'] = ((bb_lag > 0.9) | (bb_lag < 0.1)).astype(int)
        df['BB_Distance_Mean'] = abs(bb_lag - 0.5)
    
    # ============================================================
    # 8. 跨资产因子（如果有数据）
    # ============================================================
    if 'CN_US_10Y_Spread' in df.columns:
        spread_lag = df['CN_US_10Y_Spread'].shift(1)
        # 中美利差变化方向
        df['CN_US_Spread_Change_5D'] = spread_lag.diff(5)
        df['CN_US_Spread_Change_20D'] = spread_lag.diff(20)
    
    # ============================================================
    # 9. 日内模式因子
    # ============================================================
    if 'Minute_of_Day' in df.columns:
        minute_lag = df['Minute_of_Day'].shift(1)
        # 开盘/收盘效应
        df['Is_Open_Session'] = ((minute_lag >= 570) & (minute_lag <= 600)).astype(int)  # 9:30-10:00
        df['Is_Close_Session'] = ((minute_lag >= 870) & (minute_lag <= 900)).astype(int)  # 14:30-15:00
        df['Is_Lunch_Return'] = ((minute_lag >= 780) & (minute_lag <= 800)).astype(int)  # 13:00-13:20
    
    # ============================================================
    # 10. 综合制度变化检测
    # ============================================================
    # 多个异常信号同时出现
    regime_cols = []
    if 'Vol_Breakout' in df.columns:
        regime_cols.append((abs(df['Vol_Breakout']) > 1.5).astype(int))
    if 'Shock_Event' in df.columns:
        regime_cols.append(df['Shock_Event'])
    if 'Spread_Stress' in df.columns:
        regime_cols.append((abs(df['Spread_Stress']) > 1.5).astype(int))
    
    if regime_cols:
        regime_scores = pd.concat(regime_cols, axis=1).sum(axis=1)
        df['Regime_Disruption'] = (regime_scores >= 2).astype(int)
    
    print(f"[EnhancedFactor] 新增 {_count_new_cols(df)} 个因子")
    return df


def _count_consecutive(series, target):
    """计算连续出现target的次数"""
    count = 0
    for v in reversed(series.values):
        if v == target:
            count += 1
        else:
            break
    return count


def _count_new_cols(df):
    """统计新增因子数量"""
    new_patterns = [
        'Shock_Event', 'Shock_Decay', 'Cum_Shock',
        'Spread_Stress', 'Depth_Stress',
        'YC_Slope_Accel', 'YC_Flattening', 'YC_30Y_Excess', 'Butterfly_Change',
        'Vol_of_Vol', 'Vol_Persistence', 'Vol_Breakout', 'Vol_Regime_Change',
        'OI_Growth', 'OI_Acceleration', 'Volume_Surge', 'OI_Price_Divergence',
        'ADX', 'Trend_Direction',
        'RSI_Extreme', 'RSI_Distance_50', 'Consecutive', 'BB_Extreme', 'BB_Distance_Mean',
        'CN_US_Spread_Change',
        'Is_Open_Session', 'Is_Close_Session', 'Is_Lunch_Return',
        'Regime_Disruption'
    ]
    return len([c for c in df.columns if any(p in c for p in new_patterns)])
