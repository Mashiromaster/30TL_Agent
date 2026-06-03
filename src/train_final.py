# -*- coding: utf-8 -*-
# train_final.py — 完整重训练管道
# Step 1: 因子评估 → 精选Top特征
# Step 2: Optuna超参搜索
# Step 3: 集成模型训练
# Step 4: 回测对比

import pandas as pd
import numpy as np
import pickle, os, sys, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

BASE_DIR = r"D:\桌面\F_Agent"
FACTOR_FILE = os.path.join(BASE_DIR, "outputs/df_factors.pkl")
PRED_FILE = os.path.join(BASE_DIR, "outputs/df_predictions.pkl")
MODEL_FILE = os.path.join(BASE_DIR, "models/trained_model.pkl")
IMPORTANCE_FILE = os.path.join(BASE_DIR, "outputs/feature_importance.csv")

print(f"\n{'#'*60}")
print(f"#  完整重训练管道")
print(f"#  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'#'*60}")

# ==============================================
# Step 1: 因子评估 — 快速IC筛选Top50
# ==============================================
print(f"\n{'='*60}")
print(f"STEP 1: 因子评估 (Alphalens风格)")
print(f"{'='*60}")

df = pd.read_pickle(FACTOR_FILE)
df['next_close'] = df['close'].shift(-1)
df['future_close'] = df['close'].shift(-31)
df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100

# 用特征重要性预设的Top因子 + 新增因子
imp_df = pd.read_csv(IMPORTANCE_FILE) if os.path.exists(IMPORTANCE_FILE) else None

if imp_df is not None and len(imp_df) > 0:
    top_features = imp_df[imp_df['importance'] > 0]['feature'].tolist()[:50]
else:
    top_features = [c for c in df.columns if c not in {
        'date','trade_dt','ticker','close','open','high','low','volume','money','oi',
        'time','next_close','future_close','Target_Ret','Hour','Minute','Minute_of_Day'
    }][:50]

print(f"候选特征: {len(top_features)}")

# 快速IC筛选
from scipy import stats
ic_scores = []
for col in top_features:
    if col not in df.columns:
        continue
    series = df[col].shift(1)
    target = df['Target_Ret']
    mask = series.notna() & target.notna()
    if mask.sum() < 100:
        continue
    try:
        ic = stats.spearmanr(series[mask].astype(float), target[mask].astype(float))[0]
        if np.isnan(ic): ic = 0.0
        ic_scores.append({'factor': col, 'ic': abs(ic), 'ic_raw': ic})
    except:
        pass

ic_scores.sort(key=lambda x: x['ic'], reverse=True)
selected_features = [x['factor'] for x in ic_scores[:40]]
print(f"精选特征: {len(selected_features)} (Top40 by |IC|)")
top5_str = ', '.join([f"{x['factor']}({x['ic_raw']:+.4f})" for x in ic_scores[:5]])
print(f"Top 5: {top5_str}")

# ==============================================
# Step 2: 准备训练数据
# ==============================================
print(f"\n{'='*60}")
print(f"STEP 2: 准备训练数据")
print(f"{'='*60}")

df_model = df.dropna(subset=['Target_Ret']).copy()
for col in selected_features:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(0)

# Train/Val/Test split
n = len(df_model)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

train_df = df_model.iloc[:train_end].copy()
val_df = df_model.iloc[train_end:val_end].copy()
test_df = df_model.iloc[val_end:].copy()

# 12个月窗口 + 90天衰减 (最优配置)
test_start = test_df['date'].min()
cutoff = test_start - pd.Timedelta(days=12 * 30)
train_df = train_df[train_df['date'] >= cutoff].copy()

# 时间衰减权重
newest = train_df['date'].max()
age_days = (newest - train_df['date']).dt.total_seconds() / 86400
sample_weight = np.exp(-np.log(2) * age_days / 90)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Train date: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
print(f"Test date: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")

# Standardize
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(train_df[selected_features])
X_val = scaler.transform(val_df[selected_features])
X_test = scaler.transform(test_df[selected_features])
y_train = train_df['Target_Ret'].values
y_val = val_df['Target_Ret'].values
y_test = test_df['Target_Ret'].values

# ==============================================
# Step 3: Optuna 贝叶斯超参搜索
# ==============================================
print(f"\n{'='*60}")
print(f"STEP 3: Optuna 贝叶斯超参搜索 (30 trials)")
print(f"{'='*60}")

optuna_available = False
try:
    import optuna
    optuna_available = True
except ImportError:
    print("[Optuna] 未安装，使用默认参数")

import lightgbm as lgb
from lightgbm import LGBMRegressor

if optuna_available:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 48),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.5, 25, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.5, 25, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.7),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 0.8),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 80, 400),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.05, log=True),
            'objective': 'regression_l1',
            'random_state': 42, 'n_jobs': -1, 'verbose': -1,
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        y_pred = model.predict(X_val)
        ic = np.corrcoef(y_val, y_pred)[0, 1]
        return ic if not np.isnan(ic) else -1

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_params = study.best_params
    best_params['objective'] = 'regression_l1'
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    print(f"\nOptuna Best IC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
else:
    best_params = {
        'n_estimators': 200, 'learning_rate': 0.005,
        'num_leaves': 8, 'max_depth': 3,
        'lambda_l1': 15.0, 'lambda_l2': 15.0,
        'feature_fraction': 0.3, 'bagging_fraction': 0.4,
        'bagging_freq': 5, 'min_child_samples': 350,
        'min_split_gain': 0.01,
        'objective': 'regression_l1', 'random_state': 42,
        'n_jobs': -1, 'verbose': -1,
    }

# ==============================================
# Step 4: 训练最终模型
# ==============================================
print(f"\n{'='*60}")
print(f"STEP 4: 训练最终模型")
print(f"{'='*60}")

# Use discovered params
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train, sample_weight=sample_weight,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

# Test predictions
y_pred = model.predict(X_test)
test_ic = np.corrcoef(y_test, y_pred)[0, 1]
print(f"Test IC: {test_ic:.4f}")

# Regime breakdown
if 'Market_Regime' in test_df.columns:
    print(f"\n分状态IC:")
    for rid, rname in [(0, 'Normal'), (1, 'HighVol'), (2, 'Trend')]:
        mask = test_df['Market_Regime'] == rid
        if mask.sum() > 10:
            ric = np.corrcoef(y_test[mask], y_pred[mask])[0, 1]
            print(f"  {rname}: IC={ric:.4f} (n={mask.sum()})")

# ==============================================
# Step 5: XGBoost集成 (如果可用)
# ==============================================
xgboost_available = False
try:
    import xgboost as xgb
    print(f"\n{'='*60}")
    print(f"STEP 5: XGBoost集成")
    print(f"{'='*60}")

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.01,
        max_depth=4, reg_alpha=10, reg_lambda=10,
        subsample=0.5, colsample_bytree=0.4,
        objective='reg:absoluteerror',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_ic = np.corrcoef(y_test, y_pred_xgb)[0, 1]
    print(f"XGBoost Test IC: {xgb_ic:.4f}")

    # Ensemble: weighted by val IC
    lgb_val_pred = model.predict(X_val)
    xgb_val_pred = xgb_model.predict(X_val)
    lgb_val_ic = np.corrcoef(y_val, lgb_val_pred)[0, 1]
    xgb_val_ic = np.corrcoef(y_val, xgb_val_pred)[0, 1]

    w_lgb = max(0.1, lgb_val_ic) / (max(0.1, lgb_val_ic) + max(0.1, xgb_val_ic))
    w_xgb = 1 - w_lgb
    y_pred_ensemble = w_lgb * y_pred + w_xgb * y_pred_xgb
    ensemble_ic = np.corrcoef(y_test, y_pred_ensemble)[0, 1]
    print(f"Ensemble Test IC: {ensemble_ic:.4f} (LGB:{w_lgb:.2f}, XGB:{w_xgb:.2f})")
    xgboost_available = True
except ImportError:
    print(f"\n[XGBoost] 未安装，跳过集成")

# ==============================================
# Step 6: 保存
# ==============================================
print(f"\n{'='*60}")
print(f"STEP 6: 保存模型和预测")
print(f"{'='*60}")

final_pred = y_pred_ensemble if xgboost_available else y_pred

test_df_out = test_df.copy()
test_df_out['Pred_Ret'] = final_pred

output_cols = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 'Pred_Ret',
               'oi', 'volume', 'money', 'Market_Regime']
available_cols = [c for c in output_cols if c in test_df_out.columns]
test_df_out[available_cols].to_pickle(PRED_FILE)
print(f"预测已保存: {PRED_FILE} ({len(test_df_out)} 行)")

# Save model
model_data = {
    'model_base': model,
    'model_active': None,
    'scaler': scaler,
    'features': selected_features,
    'config': f'optuna_ensemble_{test_ic:.4f}',
    'test_ic': test_ic,
}
if xgboost_available:
    model_data['model_xgb'] = xgb_model
    model_data['weights'] = {'lgb': w_lgb, 'xgb': w_xgb}
    model_data['ensemble_ic'] = ensemble_ic

with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_data, f)
print(f"模型已保存: {MODEL_FILE}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance_df.to_csv(IMPORTANCE_FILE, index=False, encoding='utf-8-sig')
print(f"特征重要性 Top 10:")
for _, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.1f}")

# ==============================================
# Step 7: 回测
# ==============================================
print(f"\n{'='*60}")
print(f"STEP 7: 策略回测")
print(f"{'='*60}")

import backtest
backtest.run_process(BASE_DIR)

# Summary
print(f"\n{'#'*60}")
old_ic = 0.0388
print(f"#  重训练完成")
print(f"#  旧模型 IC: {old_ic:.4f} (12月窗口, 91特征, 手动调参)")
print(f"#  新模型 IC: {test_ic:.4f} ({len(selected_features)}特征, Optuna调参)")
if xgboost_available:
    print(f"#  集成 IC:   {ensemble_ic:.4f} (LGB+XGB)")
print(f"#  提升: +{(test_ic/old_ic-1)*100:.1f}%")
print(f"{'#'*60}")
