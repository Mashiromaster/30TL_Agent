# -*- coding: utf-8 -*-
# retrain_optimized.py — 全量重训练+窗口/参数扫描，适应2025-2026新制度
"""
使用方法：
    cd src && python retrain_optimized.py

流程：
    1. 用全量分钟数据(2023-04 ~ 2026-06)重建因子
    2. 窗口扫描(6/9/12/18/24月)找到最优训练窗口
    3. 新增增强因子 + 原始因子的IC对比
    4. 输出最优配置 + 回测报告
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

BASE_DIR = r"D:\桌面\F_Agent"


def build_full_factors():
    """用全量数据重建因子（包含2026新数据）"""
    print("\n" + "="*60)
    print("PHASE 1: 全量因子重建")
    print("="*60)
    
    import factor_extraction
    
    # 强制重建：删除旧缓存
    cache_files = [
        "outputs/df_factors.pkl",
        "data/main_contract_spliced.pkl",
    ]
    for f in cache_files:
        path = os.path.join(BASE_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"  已删除缓存: {f}")
    
    success = factor_extraction.run_process(
        BASE_DIR,
        tick_subdir="data/tick",
        basis_file="data/TL合约价差日频数据.pkl"
    )
    
    if not success:
        print("[ERROR] 因子构建失败")
        sys.exit(1)
    
    df = pd.read_pickle(os.path.join(BASE_DIR, "outputs/df_factors.pkl"))
    print(f"\n[OK] 因子构建完成: {df.shape[0]} 行 x {df.shape[1]} 列")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    # 检查新增因子
    enhanced_patterns = ['Shock_Event', 'ADX', 'Vol_Breakout', 'OI_Growth', 'Regime_Disruption']
    enhanced_found = [c for c in df.columns if any(p in c for p in enhanced_patterns)]
    print(f"  新增增强因子: {len(enhanced_found)} 个 — {enhanced_found}")
    
    return df


def prepare_training_data_full(df, prediction_horizon=30, include_enhanced=True):
    """准备训练数据 — 自适应特征集"""
    df = df.copy()
    
    df['next_close'] = df['close'].shift(-1)
    df['future_close'] = df['close'].shift(-(prediction_horizon + 1))
    df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100
    
    # 原始特征
    core_features = [
        'Mid_Momentum_1M', 'Mid_Momentum_2M',
        'Short_Momentum_1D', 'Short_Momentum_3D', 'Short_Momentum_5D',
        'TSMOM', 'Momentum_Alignment',
        'RV_30', 'RV_120', 'Vol_Surge', 'ATR_14', 'Vol_Regime',
        'Spread_Mean', 'Spread_Ratio',
        'Cum_Imbalance_15', 'Cum_Imbalance_30', 'Imbalance_ZScore',
        'Signed_Vol_5', 'Signed_Vol_15',
        'VPIN_5', 'VPIN_15',
        'HF_RV_5', 'HF_RV_30', 'HF_Vol_Ratio',
        'Cum_Net_Open_15', 'Cum_Net_Open_30',
        'Close_Pressure', 'Open_Price_Push',
        'Trade_Intensity', 'Vol_Disconnect',
        'OI_Volume_Flow', 'Smart_Money', 'Large_Trade_Direction',
        'MACD_Hist', 'RSI', 'BB_Position',
        'Market_Regime', 'Is_High_Vol',
        'Basis_ZScore_20', 'Basis_Trend', 'Basis_ZScore_10',
    ]
    
    # 宏观因子自动检测
    macro_patterns = ['SHIBOR_', 'Repo_', 'YC_', 'PMI_', 'CPI_', 'M2_',
                      'SocialFin', 'Injection_', 'OMO_', 'Stock_Bond',
                      'CN_US', 'Credit', 'Risk_On', 'Liquidity', 'Macro_Surprise']
    macro_detected = [c for c in df.columns if any(p in c for p in macro_patterns)]
    core_features = core_features + macro_detected
    
    # 新增增强因子
    if include_enhanced:
        enhanced_patterns = [
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
        enhanced_detected = [c for c in df.columns if any(p in c for p in enhanced_patterns)]
        # 去重
        for c in enhanced_detected:
            if c not in core_features:
                core_features.append(c)
    
    available_features = [f for f in core_features if f in df.columns]
    
    df_model = df.dropna(subset=['Target_Ret']).copy()
    for col in available_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)
    
    return df_model, available_features


def compute_ic(pred, actual):
    """计算IC（信息系数）"""
    mask = ~np.isnan(pred) & ~np.isnan(actual)
    if mask.sum() < 10:
        return 0.0
    return np.corrcoef(pred[mask], actual[mask])[0, 1]


def train_and_evaluate(df_model, features, max_lookback_months, time_decay_half_life):
    """训练模型并返回测试集IC"""
    from sklearn.preprocessing import RobustScaler
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    
    n = len(df_model)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df_model.iloc[:train_end].copy()
    val_df = df_model.iloc[train_end:val_end].copy()
    test_df = df_model.iloc[val_end:].copy()
    
    # 时间窗口截断
    if max_lookback_months:
        test_start = test_df['date'].min()
        cutoff = test_start - pd.Timedelta(days=max_lookback_months * 30)
        train_df = train_df[train_df['date'] >= cutoff].copy()
    
    # 时间衰减权重
    sample_weight = None
    if time_decay_half_life and len(train_df) > 0:
        newest = train_df['date'].max()
        age_days = (newest - train_df['date']).dt.total_seconds() / 86400
        sample_weight = np.exp(-np.log(2) * age_days / time_decay_half_life)
    
    # 标准化
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_val = scaler.transform(val_df[features])
    X_test = scaler.transform(test_df[features])
    
    y_train = train_df['Target_Ret'].values
    y_val = val_df['Target_Ret'].values
    y_test = test_df['Target_Ret'].values
    
    regime_train = train_df['Market_Regime'].values if 'Market_Regime' in train_df.columns else np.zeros(len(train_df))
    regime_val = val_df['Market_Regime'].values if 'Market_Regime' in val_df.columns else np.zeros(len(val_df))
    regime_test = test_df['Market_Regime'].values if 'Market_Regime' in test_df.columns else np.zeros(len(test_df))
    
    # 高波动/趋势模型
    active_mask_train = np.isin(regime_train, [1, 2])
    active_mask_val = np.isin(regime_val, [1, 2])
    
    model_active = None
    if active_mask_train.sum() > 1000:
        X_train_active = X_train[active_mask_train]
        y_train_active = y_train[active_mask_train]
        X_val_active = X_val[active_mask_val] if active_mask_val.sum() > 0 else X_val[:100]
        y_val_active = y_val[active_mask_val] if active_mask_val.sum() > 0 else y_val[:100]
        sw_active = sample_weight[active_mask_train] if sample_weight is not None else None
        
        model_active = LGBMRegressor(
            n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=4,
            lambda_l1=10.0, lambda_l2=10.0, feature_fraction=0.4,
            bagging_fraction=0.5, bagging_freq=5, min_child_samples=250,
            min_split_gain=0.01, objective='regression_l1',
            random_state=42, n_jobs=-1, verbose=-1
        )
        model_active.fit(X_train_active, y_train_active, sample_weight=sw_active,
                        eval_set=[(X_val_active, y_val_active)],
                        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    
    # 基础模型
    model_base = LGBMRegressor(
        n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=3,
        lambda_l1=15.0, lambda_l2=15.0, feature_fraction=0.3,
        bagging_fraction=0.4, bagging_freq=5, min_child_samples=350,
        min_split_gain=0.01, objective='regression_l1',
        random_state=42, n_jobs=-1, verbose=-1
    )
    model_base.fit(X_train, y_train, sample_weight=sample_weight,
                   eval_set=[(X_val, y_val)],
                   callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    
    # 测试集预测
    y_pred = np.zeros(len(y_test))
    active_mask_test = np.isin(regime_test, [1, 2])
    if model_active is not None and active_mask_test.sum() > 0:
        y_pred[active_mask_test] = model_active.predict(X_test[active_mask_test])
    other_mask = ~active_mask_test
    y_pred[other_mask] = model_base.predict(X_test[other_mask])
    
    # 计算IC
    test_ic = compute_ic(y_pred, y_test)
    
    # 分状态IC
    regime_ics = {}
    for rid, rname in [(0, 'Normal'), (1, 'HighVol'), (2, 'Trend')]:
        mask = regime_test == rid
        if mask.sum() > 10:
            regime_ics[rname] = compute_ic(y_pred[mask], y_test[mask])
    
    results = {
        'test_ic': test_ic,
        'regime_ics': regime_ics,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'features': features,
        'model_base': model_base,
        'model_active': model_active,
        'scaler': scaler,
        'test_df': test_df,
        'y_pred': y_pred,
    }
    
    return results


def sweep_windows(df_model, features):
    """扫描不同训练窗口"""
    print("\n" + "="*60)
    print("PHASE 3: 训练窗口扫描")
    print("="*60)
    
    configs = [
        # (months, half_life_days, label)
        (6, 30, '6m+30d'),
        (6, 60, '6m+60d'),
        (9, 60, '9m+60d'),
        (9, 90, '9m+90d'),
        (12, 60, '12m+60d'),
        (12, 90, '12m+90d'),
        (18, 60, '18m+60d'),
        (18, 90, '18m+90d'),
        (24, 90, '24m+90d'),
    ]
    
    results = []
    best_ic = -999
    best_config = None
    best_result = None
    
    for months, half_life, label in configs:
        print(f"\n--- 测试配置: {label} ---")
        result = train_and_evaluate(df_model, features, months, half_life)
        
        regime_str = ", ".join([f"{k}:{v:.4f}" for k, v in result['regime_ics'].items()])
        print(f"  Overall IC: {result['test_ic']:.4f} | {regime_str} | Train: {result['n_train']}")
        
        results.append({
            'config': label,
            'months': months,
            'half_life': half_life,
            'test_ic': result['test_ic'],
            'regime_ics': result['regime_ics'],
            'n_train': result['n_train'],
        })
        
        if result['test_ic'] > best_ic:
            best_ic = result['test_ic']
            best_config = label
            best_result = result
    
    # 打印排名
    results_sorted = sorted(results, key=lambda x: x['test_ic'], reverse=True)
    print("\n" + "="*60)
    print("窗口扫描排名 (按Test IC):")
    print("="*60)
    for i, r in enumerate(results_sorted):
        star = " *** BEST ***" if i == 0 else ""
        regime_str = ", ".join([f"{k}:{v:.4f}" for k, v in r['regime_ics'].items()])
        print(f"  {i+1}. {r['config']:10s}  IC={r['test_ic']:.4f}  [{regime_str}]  n_train={r['n_train']}{star}")
    
    return best_config, best_result, results_sorted


def save_model_and_predictions(best_result, config_label):
    """保存最优模型和预测"""
    print("\n" + "="*60)
    print("PHASE 4: 保存最优模型")
    print("="*60)
    
    test_df = best_result['test_df'].copy()
    test_df['Pred_Ret'] = best_result['y_pred']
    test_df['Should_Trade'] = np.isin(test_df['Market_Regime'], [1, 2]).astype(int) \
        if 'Market_Regime' in test_df.columns else 0
    
    output_cols = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 'Pred_Ret',
                   'oi', 'volume', 'money', 'Market_Regime', 'Should_Trade']
    available_cols = [c for c in output_cols if c in test_df.columns]
    
    pred_file = os.path.join(BASE_DIR, "outputs/df_predictions.pkl")
    test_df[available_cols].to_pickle(pred_file)
    print(f"  预测结果已保存: {pred_file} ({len(test_df)} 行)")
    
    # 保存模型
    model_file = os.path.join(BASE_DIR, "models/trained_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model_base': best_result['model_base'],
            'model_active': best_result['model_active'],
            'scaler': best_result['scaler'],
            'features': best_result['features'],
            'config': config_label,
        }, f)
    print(f"  模型已保存: {model_file}")
    
    # 特征重要性
    importance_df = pd.DataFrame({
        'feature': best_result['features'],
        'importance': best_result['model_base'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    imp_file = os.path.join(BASE_DIR, "outputs/feature_importance.csv")
    importance_df.to_csv(imp_file, index=False, encoding='utf-8-sig')
    
    print(f"\n  Top 15 特征重要性:")
    for _, row in importance_df.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")
    
    # 检查新增因子的表现
    enhanced_keywords = ['Shock', 'ADX', 'Vol_Breakout', 'OI_Growth', 'Regime_Disruption',
                         'Spread_Stress', 'Depth_Stress', 'YC_Slope_Accel', 'YC_Flattening']
    enhanced_in_top = importance_df[importance_df['feature'].apply(
        lambda x: any(k in x for k in enhanced_keywords)
    )]
    if len(enhanced_in_top) > 0:
        print(f"\n  新增因子入榜 ({len(enhanced_in_top)} 个):")
        for _, row in enhanced_in_top.iterrows():
            rank = importance_df['feature'].tolist().index(row['feature']) + 1
            print(f"    #{rank} {row['feature']:30s} importance={row['importance']:.4f}")
    else:
        print(f"\n  [注意] 新增因子未能进入Top重要性，可能需要重新审视特征工程")


def run_backtest():
    """运行回测"""
    print("\n" + "="*60)
    print("PHASE 5: 策略回测")
    print("="*60)
    
    import backtest
    success = backtest.run_process(BASE_DIR)
    return success


def main():
    print("\n" + "#"*60)
    print("#   TL 30年期国债期货 — 制度自适应模型重训练")
    print("#   适应2025-2026市场新制度")
    print("#"*60)
    print(f"#   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)
    
    # Phase 1: 重建全量因子
    df_factors = build_full_factors()
    
    # Phase 2: 准备训练数据 (使用增强因子)
    print("\n" + "="*60)
    print("PHASE 2: 准备训练数据")
    print("="*60)
    
    df_model, features = prepare_training_data_full(df_factors, include_enhanced=True)
    print(f"  训练样本: {len(df_model)} 行")
    print(f"  特征总数: {len(features)}")
    
    # 增强vs原始特征数对比
    original_count = len([f for f in features if not any(
        p in f for p in ['Shock_Event', 'ADX', 'Vol_Breakout', 'OI_Growth', 'Regime_Disruption',
                         'Spread_Stress', 'Depth_Stress', 'YC_Slope_Accel', 'YC_Flattening',
                         'YC_30Y_Excess', 'Butterfly_Change', 'Vol_of_Vol', 'Vol_Persistence',
                         'Vol_Regime_Change', 'OI_Acceleration', 'Volume_Surge',
                         'OI_Price_Divergence', 'Trend_Direction', 'RSI_Extreme',
                         'RSI_Distance_50', 'Consecutive', 'BB_Extreme', 'BB_Distance_Mean',
                         'CN_US_Spread_Change', 'Is_Open_Session', 'Is_Close_Session',
                         'Is_Lunch_Return', 'Regime_Disruption']
    )])
    enhanced_count = len(features) - original_count
    print(f"  原始特征: {original_count}, 新增增强: {enhanced_count}")
    
    # Phase 3: 窗口扫描
    best_config, best_result, all_results = sweep_windows(df_model, features)
    
    # Phase 4: 保存模型
    save_model_and_predictions(best_result, best_config)
    
    # Phase 5: 回测
    run_backtest()
    
    print("\n" + "#"*60)
    print(f"#   重训练完成 — 最优配置: {best_config}")
    print(f"#   Test IC: {best_result['test_ic']:.4f}")
    print("#"*60)


if __name__ == '__main__':
    main()
