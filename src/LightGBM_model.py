# -*- coding: utf-8 -*-
# LightGBM_model.py (V3.0 - 优化版)

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

def prepare_training_data(df, prediction_horizon=30):
    """准备训练数据"""
    df = df.copy()
    
    df['next_close'] = df['close'].shift(-1)
    df['future_close'] = df['close'].shift(-(prediction_horizon + 1))
    df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100
    
        # 精选核心因子（基于重要性分析）
    core_features = [
        # === 动量因子 ===
        'Mid_Momentum_1M', 'Mid_Momentum_2M',
        'Short_Momentum_1D', 'Short_Momentum_3D', 'Short_Momentum_5D',
        'TSMOM', 'Momentum_Alignment',
        
        # === 波动率因子 ===
        'RV_30', 'RV_120', 'Vol_Surge', 'ATR_14', 'Vol_Regime',
        
        # === 微观结构因子 ===
        'Spread_Mean', 'Spread_Ratio',
        'Cum_Imbalance_15', 'Cum_Imbalance_30', 'Imbalance_ZScore',
        'Signed_Vol_5', 'Signed_Vol_15',
        'VPIN_5', 'VPIN_15',
        'HF_RV_5', 'HF_RV_30', 'HF_Vol_Ratio',
        'Cum_Net_Open_15', 'Cum_Net_Open_30',
        'Close_Pressure', 'Open_Price_Push',
        'Trade_Intensity', 'Vol_Disconnect',
        
        # === 量价因子 ===
        'OI_Volume_Flow', 'Smart_Money', 'Large_Trade_Direction',
        
        # === 技术因子 ===
        'MACD_Hist', 'RSI', 'BB_Position',
        
        # === 市场状态 ===
        'Market_Regime', 'Is_High_Vol',
        
        # === 基差因子===
        'Basis_ZScore_20', 'Basis_Trend','Basis_ZScore_10',
    ]
    
    # Auto-detect macro / funding factor columns from DataFrame
    macro_patterns = ['SHIBOR_', 'Repo_', 'YC_', 'PMI_', 'CPI_', 'M2_',
                      'SocialFin', 'Injection_', 'OMO_', 'Stock_Bond',
                      'CN_US', 'Credit', 'Risk_On', 'Liquidity', 'Macro_Surprise']
    macro_detected = [c for c in df.columns if any(p in c for p in macro_patterns)]
    if macro_detected:
        core_features = core_features + macro_detected
        print(f"[Model] 自动检测到 {len(macro_detected)} 个宏观因子")

    available_features = [f for f in core_features if f in df.columns]
    
    micro_features = [f for f in available_features if any(
        keyword in f for keyword in ['Spread', 'Imbalance', 'VPIN', 'HF_', 'Signed', 'Trade_', 'Open_', 'Close_', 'Cum_', 'Vol_Disconnect']
    )]
    
    print(f"[Model] 可用特征数: {len(available_features)}")
    print(f"[Model] 其中微观结构特征: {len(micro_features)}")

    macro_model_features = [f for f in available_features if any(p in f for p in macro_patterns)]
    if macro_model_features:
        print(f"[Model] 其中宏观特征: {len(macro_model_features)}")    
    df_model = df.dropna(subset=['Target_Ret']).copy()
    
    for col in available_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)
    
    return df_model, available_features


def train_model(X_train, y_train, X_val, y_val, model_type='base'):
    """训练模型 - 更强正则化"""
    if model_type == 'highvol':
        model = LGBMRegressor(
            n_estimators=80,
            learning_rate=0.005,
            num_leaves=6,
            max_depth=4,
            lambda_l1=15.0,
            lambda_l2=15.0,
            feature_fraction=0.3,
            bagging_fraction=0.5,
            bagging_freq=5,
            min_child_samples=300,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    else:
        model = LGBMRegressor(
            n_estimators=60,
            learning_rate=0.003,
            num_leaves=4,
            max_depth=2,
            lambda_l1=20.0,
            lambda_l2=20.0,
            feature_fraction=0.25,
            bagging_fraction=0.4,
            bagging_freq=5,
            min_child_samples=500,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[]
    )
    
    return model


def run_process(base_dir):
    print("\n" + "="*50)
    print("STEP 3: 模型训练")
    print("="*50)
    
    FACTOR_FILE = os.path.join(base_dir, "outputs/df_factors.pkl")
    PRED_FILE = os.path.join(base_dir, "outputs/df_predictions.pkl")
    MODEL_FILE = os.path.join(base_dir, "models/trained_model.pkl")
    IMPORTANCE_FILE = os.path.join(base_dir, "outputs/feature_importance.csv")
    
    if not os.path.exists(FACTOR_FILE):
        print(f"[Model ERROR] 找不到因子文件: {FACTOR_FILE}")
        return False
    
    df = pd.read_pickle(FACTOR_FILE)
    df_model, features = prepare_training_data(df)
    
    print(f"[Model] 总样本数: {len(df_model)}, 特征数: {len(features)}")
    
    n = len(df_model)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df_model.iloc[:train_end].copy()
    val_df = df_model.iloc[train_end:val_end].copy()
    test_df = df_model.iloc[val_end:].copy()
    
    print(f"[Model] 训练集: {len(train_df)} ({len(train_df)/n:.1%})")
    print(f"[Model] 验证集: {len(val_df)} ({len(val_df)/n:.1%})")
    print(f"[Model] 测试集: {len(test_df)} ({len(test_df)/n:.1%})")
    
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
    
    # === 训练高波动市 + 趋势市模型 ===
    print("\n[Model] 训练高波动/趋势市模型...")
    active_mask_train = np.isin(regime_train, [1, 2])
    active_mask_val = np.isin(regime_val, [1, 2])
    
    model_active = None
    if active_mask_train.sum() > 1000:
        X_train_active = X_train[active_mask_train]
        y_train_active = y_train[active_mask_train]
        X_val_active = X_val[active_mask_val] if active_mask_val.sum() > 0 else X_val[:100]
        y_val_active = y_val[active_mask_val] if active_mask_val.sum() > 0 else y_val[:100]
        
        model_active = train_model(X_train_active, y_train_active, X_val_active, y_val_active, 'highvol')
        
        train_pred_active = model_active.predict(X_train_active)
        train_ic_active = np.corrcoef(y_train_active, train_pred_active)[0, 1]
        print(f"  高波动/趋势市训练集 IC: {train_ic_active:.4f}")
    
    # === 训练基础模型 ===
    print("\n[Model] 训练基础模型...")
    model_base = train_model(X_train, y_train, X_val, y_val, 'base')
    
    y_val_pred = model_base.predict(X_val)
    val_ic = np.corrcoef(y_val, y_val_pred)[0, 1]
    print(f"  基础模型验证集 IC: {val_ic:.4f}")
    
    # === 测试集预测 ===
    print("\n[Model] 测试集预测...")
    y_test_pred = np.zeros(len(y_test))
    
    # 高波动/趋势市用专用模型
    active_mask_test = np.isin(regime_test, [1, 2])
    if model_active is not None and active_mask_test.sum() > 0:
        y_test_pred[active_mask_test] = model_active.predict(X_test[active_mask_test])
    
    # 其他市场用基础模型
    other_mask_test = ~active_mask_test
    y_test_pred[other_mask_test] = model_base.predict(X_test[other_mask_test])
    
    test_ic = np.corrcoef(y_test, y_test_pred)[0, 1]
    print(f"\n[Model] 测试集整体 IC: {test_ic:.4f}")
    
    print(f"\n[Model] 分状态 IC 分析:")
    for regime_id in [0, 1, 2]:
        mask = (regime_test == regime_id)
        if mask.sum() > 10:
            regime_ic = np.corrcoef(y_test[mask], y_test_pred[mask])[0, 1]
            regime_name = ['正常', '高波动', '趋势'][regime_id]
            print(f"  - {regime_name}: IC={regime_ic:.4f}, 样本数={mask.sum()}")
    
    # 保存预测
    test_df_out = test_df.copy()
    test_df_out['Pred_Ret'] = y_test_pred
    test_df_out['Should_Trade'] = np.isin(test_df_out['Market_Regime'], [1, 2]).astype(int)
    
    output_cols = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 'Pred_Ret', 
                   'oi', 'volume', 'money', 'Market_Regime', 'Should_Trade']
    available_cols = [c for c in output_cols if c in test_df_out.columns]
    test_df_out[available_cols].to_pickle(PRED_FILE)
    
    print(f"\n[Model] 测试集预测已保存: {PRED_FILE}")
    
    # 保存模型
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model_active': model_active,
            'model_base': model_base,
            'scaler': scaler,
            'features': features
        }, f)
    
    # 特征重要性
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model_base.feature_importances_
    }).sort_values('importance', ascending=False)
    importance_df.to_csv(IMPORTANCE_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n[Model] Top 15 重要特征:")
    print(importance_df.head(15).to_string(index=False))
    
    return True