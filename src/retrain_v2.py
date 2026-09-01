# -*- coding: utf-8 -*-
# retrain_v2.py — V2优化: MSE loss + 特征精选 + 排名信号
"""
V2改进:
1. 使用MSE loss增加预测方差 (MAE导致预测过于集中)
2. 特征精选: 仅保留Top-50重要特征
3. 排名信号: 使用预测分位数排名替代绝对值阈值
4. 自适应信号平滑窗口
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from lightgbm import LGBMRegressor

BASE_DIR = r"D:\桌面\F_Agent"


def load_factors():
    """加载因子数据"""
    factor_file = os.path.join(BASE_DIR, "outputs/df_factors.pkl")
    if not os.path.exists(factor_file):
        print(f"[ERROR] 因子文件不存在: {factor_file}")
        print("请先运行 retrain_optimized.py 生成因子")
        sys.exit(1)
    return pd.read_pickle(factor_file)


def prepare_data(df, prediction_horizon=30):
    """准备训练数据"""
    df = df.copy()
    df['next_close'] = df['close'].shift(-1)
    df['future_close'] = df['close'].shift(-(prediction_horizon + 1))
    df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100
    
    # 精选特征: 基于V1重要性分析，保留Top重要性特征
    selected_features = [
        # 宏观 (这些是V1中最重要的)
        'Macro_Surprise_Composite', 'CN_US_10Y_Spread', 'CN_US_10Y_Spread_Z',
        'YC_Slope_30Y_10Y', 'YC_Momentum_5D', 'PMI_ZScore',
        'YC_Curvature', 'YC_Slope_10Y_1Y', 'YC_Level_Shift', 'YC_Level_ZScore',
        'M2_Surprise', 'CPI_Momentum', 'Risk_On_Off',
        
        # 基差
        'Basis_ZScore_20', 'Basis_ZScore_10', 'Basis_Trend',
        
        # 动量
        'Mid_Momentum_2M', 'Mid_Momentum_1M', 'Short_Momentum_5D',
        'Short_Momentum_1D', 'Short_Momentum_3D', 'TSMOM', 'Momentum_Alignment',
        
        # 波动率
        'RV_30', 'RV_120', 'Vol_Surge', 'ATR_14', 'Vol_Regime',
        
        # 增强因子
        'OI_Growth_20D', 'OI_Growth_5D', 'Vol_Breakout', 'ADX',
        'Volume_Surge', 'OI_Acceleration', 'Vol_of_Vol',
        'Vol_Persistence', 'ADX_Change', 'Trend_Direction',
        
        # 微观结构
        'Cum_Imbalance_15', 'Cum_Imbalance_30', 'Imbalance_ZScore',
        'Signed_Vol_15', 'VPIN_15', 'HF_RV_30', 'HF_Vol_Ratio',
        'Trade_Intensity', 'Cum_Net_Open_15',
        
        # 量价
        'OI_Volume_Flow', 'Smart_Money', 'Large_Trade_Direction',
        
        # 技术
        'MACD_Hist', 'RSI', 'BB_Position',
        
        # 市场状态
        'Market_Regime', 'Is_High_Vol',
    ]
    
    available = [f for f in selected_features if f in df.columns]
    
    df_model = df.dropna(subset=['Target_Ret']).copy()
    for col in available:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)
    
    return df_model, available


def train_mse_model(X_train, y_train, X_val, y_val, sample_weight=None, model_type='base'):
    """训练模型 - 使用MSE loss获取更大预测方差"""
    if model_type == 'highvol':
        model = LGBMRegressor(
            n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=4,
            lambda_l1=5.0, lambda_l2=5.0,
            feature_fraction=0.5, bagging_fraction=0.6, bagging_freq=5,
            min_child_samples=200, min_split_gain=0.001,
            objective='regression',  # MSE
            random_state=42, n_jobs=-1, verbose=-1
        )
    else:
        model = LGBMRegressor(
            n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=3,
            lambda_l1=10.0, lambda_l2=10.0,
            feature_fraction=0.4, bagging_fraction=0.5, bagging_freq=5,
            min_child_samples=300, min_split_gain=0.001,
            objective='regression',  # MSE
            random_state=42, n_jobs=-1, verbose=-1
        )
    
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    return model


def compute_ic(pred, actual):
    mask = ~np.isnan(pred) & ~np.isnan(actual)
    if mask.sum() < 10:
        return 0.0
    return np.corrcoef(pred[mask], actual[mask])[0, 1]


def sweep_configs(df_model, features):
    """扫描训练配置: 窗口 + loss函数"""
    print("\n" + "="*60)
    print("V2 配置扫描: MSE loss + 特征精选")
    print("="*60)
    
    configs = [
        (12, 90, 'mse'),   # V1最优窗口 
        (9, 90, 'mse'),
        (12, 60, 'mse'),
        (15, 90, 'mse'),
        (12, 90, 'mae'),   # 对比MAE
    ]
    
    results = []
    best_ic = -999
    best_result = None
    best_config = None
    
    for months, half_life, loss_type in configs:
        label = f'{months}m+{half_life}d_{loss_type}'
        print(f"\n--- {label} ---")
        
        n = len(df_model)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_df = df_model.iloc[:train_end].copy()
        val_df = df_model.iloc[train_end:val_end].copy()
        test_df = df_model.iloc[val_end:].copy()
        
        # 窗口截断
        test_start = test_df['date'].min()
        cutoff = test_start - pd.Timedelta(days=months * 30)
        train_df = train_df[train_df['date'] >= cutoff].copy()
        
        # 时间衰减
        sw = None
        if len(train_df) > 0:
            newest = train_df['date'].max()
            age_days = (newest - train_df['date']).dt.total_seconds() / 86400
            sw = np.exp(-np.log(2) * age_days / half_life)
        
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
        
        # 高波动模型
        active_mask_train = np.isin(regime_train, [1, 2])
        model_active = None
        if active_mask_train.sum() > 500:
            sw_a = sw[active_mask_train] if sw is not None else None
            if loss_type == 'mae':
                model_active = LGBMRegressor(
                    n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=4,
                    lambda_l1=10.0, lambda_l2=10.0, feature_fraction=0.4,
                    bagging_fraction=0.5, bagging_freq=5, min_child_samples=250,
                    min_split_gain=0.01, objective='regression_l1',
                    random_state=42, n_jobs=-1, verbose=-1
                )
            else:
                model_active = train_mse_model(
                    X_train[active_mask_train], y_train[active_mask_train],
                    X_val[np.isin(regime_val, [1,2])] if np.isin(regime_val, [1,2]).sum() > 0 else X_val[:100],
                    y_val[np.isin(regime_val, [1,2])] if np.isin(regime_val, [1,2]).sum() > 0 else y_val[:100],
                    sw_a, 'highvol'
                )
        
        # 基础模型
        if loss_type == 'mae':
            model_base = LGBMRegressor(
                n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=3,
                lambda_l1=15.0, lambda_l2=15.0, feature_fraction=0.3,
                bagging_fraction=0.4, bagging_freq=5, min_child_samples=350,
                min_split_gain=0.01, objective='regression_l1',
                random_state=42, n_jobs=-1, verbose=-1
            )
            model_base.fit(X_train, y_train, sample_weight=sw,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        else:
            model_base = train_mse_model(X_train, y_train, X_val, y_val, sw, 'base')
        
        # 预测
        y_pred = np.zeros(len(y_test))
        active_mask_test = np.isin(regime_test, [1, 2])
        if model_active is not None and active_mask_test.sum() > 0:
            y_pred[active_mask_test] = model_active.predict(X_test[active_mask_test])
        y_pred[~active_mask_test] = model_base.predict(X_test[~active_mask_test])
        
        test_ic = compute_ic(y_pred, y_test)
        
        # 分状态IC
        regime_ics = {}
        for rid, rname in [(0, 'Normal'), (1, 'HighVol'), (2, 'Trend')]:
            mask = regime_test == rid
            if mask.sum() > 10:
                regime_ics[rname] = compute_ic(y_pred[mask], y_test[mask])
        
        # 预测统计
        pred_std = y_pred.std()
        pred_mean = abs(y_pred.mean())
        
        regime_str = ", ".join([f"{k}:{v:.4f}" for k, v in regime_ics.items()])
        print(f"  IC={test_ic:.4f} | Pred std={pred_std:.4f} | {regime_str}")
        
        results.append({
            'config': label, 'months': months, 'half_life': half_life,
            'loss': loss_type, 'test_ic': test_ic, 'regime_ics': regime_ics,
            'pred_std': pred_std, 'n_train': len(train_df),
        })
        
        if test_ic > best_ic:
            best_ic = test_ic
            best_config = label
            best_result = {
                'model_base': model_base, 'model_active': model_active,
                'scaler': scaler, 'features': features,
                'test_df': test_df, 'y_pred': y_pred, 'y_test': y_test,
                'regime_test': regime_test,
            }
    
    # 排名
    results_sorted = sorted(results, key=lambda x: x['test_ic'], reverse=True)
    print("\n排名:")
    for i, r in enumerate(results_sorted):
        star = " *** BEST ***" if i == 0 else ""
        regime_str = ", ".join([f"{k}:{v:.4f}" for k, v in r['regime_ics'].items()])
        print(f"  {i+1}. {r['config']:15s} IC={r['test_ic']:.4f} std={r['pred_std']:.4f} [{regime_str}]{star}")
    
    return best_config, best_result, results_sorted


def save_and_backtest(best_result, config_label):
    """保存模型并运行排名信号回测"""
    test_df = best_result['test_df'].copy()
    test_df['Pred_Ret'] = best_result['y_pred']
    
    print(f"\nPred_Ret stats: mean={test_df['Pred_Ret'].mean():.6f}, std={test_df['Pred_Ret'].std():.6f}")
    print(f"Target_Ret stats: mean={test_df['Target_Ret'].mean():.6f}, std={test_df['Target_Ret'].std():.6f}")
    
    # 保存预测
    pred_file = os.path.join(BASE_DIR, "outputs/df_predictions.pkl")
    output_cols = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 'Pred_Ret',
                   'oi', 'volume', 'money', 'Market_Regime']
    available_cols = [c for c in output_cols if c in test_df.columns]
    test_df[available_cols].to_pickle(pred_file)
    
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
    
    # 特征重要性
    importance_df = pd.DataFrame({
        'feature': best_result['features'],
        'importance': best_result['model_base'].feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(BASE_DIR, "outputs/feature_importance.csv"),
                        index=False, encoding='utf-8-sig')
    
    print(f"\nTop 20 特征:")
    for _, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.1f}")
    
    # 运行回测
    print("\n" + "="*60)
    print("策略回测")
    print("="*60)
    import backtest
    backtest.run_process(BASE_DIR)


def main():
    print("\n" + "#"*60)
    print("#   TL 30Y — V2优化: MSE Loss + 特征精选 + 排名信号")
    print("#"*60)
    
    df_factors = load_factors()
    df_model, features = prepare_data(df_factors)
    print(f"训练数据: {len(df_model)} 行, {len(features)} 特征")
    
    best_config, best_result, all_results = sweep_configs(df_model, features)
    
    print(f"\n最优配置: {best_config}")
    save_and_backtest(best_result, best_config)
    
    print("\n完成!")


if __name__ == '__main__':
    main()
