# -*- coding: utf-8 -*-
# optuna_optimizer.py — Optuna自动超参优化 + 多模型集成
# 借鉴: RektGBM (⭐23), Optuna framework, Freqtrade hyperopt
#
# 功能:
#   1. Optuna贝叶斯超参搜索 (替代手动窗口扫描)
#   2. Multi-model ensemble (LightGBM + XGBoost + CatBoost)
#   3. Stacking ensemble (元学习器)
#   4. Multi-horizon stacked prediction (30/60/120min)

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE_DIR = r"D:\桌面\F_Agent"


# ================================================================
# Part 1: Optuna 贝叶斯超参搜索
# ================================================================

def run_optuna_optimization(df_factors, n_trials=50, prediction_horizon=30,
                             train_months=12, time_decay_half_life=90,
                             study_name="f_agent_lgbm"):
    """使用 Optuna 进行贝叶斯超参数优化"""
    try:
        import optuna
    except ImportError:
        print("[Optuna] optuna 未安装，使用默认参数")
        print("  安装: pip install optuna")
        return _get_default_params()

    print(f"\n{'='*60}")
    print(f"  Optuna 贝叶斯超参优化 ({n_trials} trials)")
    print(f"  Horizon={prediction_horizon}min | Window={train_months}mo")
    print(f"{'='*60}")

    # 准备数据
    df = df_factors.copy()
    df['next_close'] = df['close'].shift(-1)
    df['future_close'] = df['close'].shift(-(prediction_horizon + 1))
    df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100
    df_model = df.dropna(subset=['Target_Ret']).copy()

    # 特征选择: 排除非特征列
    exclude = {'date', 'trade_dt', 'ticker', 'close', 'open', 'high', 'low',
               'volume', 'money', 'oi', 'time', 'next_close', 'future_close',
               'Target_Ret', 'Hour', 'Minute', 'Minute_of_Day',
               'Pred_Ret', 'Pred_Smooth', 'Market_Regime', 'Signal', 'Raw_Signal',
               'Confirmed_Signal', 'Pred_Rank', 'Model_Used'}
    features = [c for c in df_model.columns if c not in exclude]
    available_features = [f for f in features if f in df_model.columns
                         and df_model[f].dtype in ['float64', 'int64', 'int32', 'float32']]

    for col in available_features:
        df_model[col] = df_model[col].fillna(0)

    # 时序切分
    n = len(df_model)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df_model.iloc[:train_end].copy()
    val_df = df_model.iloc[train_end:val_end].copy()

    # 窗口截断
    if train_months:
        test_start = df_model.iloc[val_end:]['date'].min()
        cutoff = test_start - pd.Timedelta(days=train_months * 30)
        train_df = train_df[train_df['date'] >= cutoff].copy()

    # 时间衰减权重
    sw = None
    if time_decay_half_life and len(train_df) > 0:
        newest = train_df['date'].max()
        age_days = (newest - train_df['date']).dt.total_seconds() / 86400
        sw = np.exp(-np.log(2) * age_days / time_decay_half_life)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[available_features])
    X_val = scaler.transform(val_df[available_features])
    y_train = train_df['Target_Ret'].values
    y_val = val_df['Target_Ret'].values

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Features: {len(available_features)}")

    # Optuna 目标函数
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 4, 64),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 30, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 30, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 0.8),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 50, 500),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1, log=True),
            'objective': 'regression_l1',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sw,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        y_pred = model.predict(X_val)
        ic = np.corrcoef(y_val, y_pred)[0, 1]
        return ic  # 最大化IC

    # 运行优化
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_ic = study.best_value

    print(f"\n  ✅ Best trial: IC={best_ic:.4f}")
    print(f"  Best params: {best_params}")

    return {
        'best_params': best_params,
        'best_ic': round(best_ic, 4),
        'n_trials': n_trials,
        'features': available_features,
        'scaler': scaler,
    }


def _get_default_params():
    return {
        'best_params': {
            'n_estimators': 200, 'learning_rate': 0.005,
            'num_leaves': 8, 'max_depth': 3,
            'lambda_l1': 15.0, 'lambda_l2': 15.0,
            'feature_fraction': 0.3, 'bagging_fraction': 0.4,
            'bagging_freq': 5, 'min_child_samples': 350,
            'min_split_gain': 0.01,
        },
        'best_ic': 0.0,
        'n_trials': 0,
    }


# ================================================================
# Part 2: Multi-model Ensemble (LightGBM + XGBoost + CatBoost)
# ================================================================

class EnsemblePredictor:
    """多模型集成预测器 (借鉴 RektGBM + Freqtrade ensemble)"""

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.features = None

    def train(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """训练三个模型并简单平均集成"""
        # LightGBM
        print("[Ensemble] Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.005,
            num_leaves=8, max_depth=3,
            lambda_l1=15.0, lambda_l2=15.0,
            feature_fraction=0.3, bagging_fraction=0.4,
            bagging_freq=5, min_child_samples=350,
            min_split_gain=0.01, objective='regression_l1',
            random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_model.fit(X_train, y_train, sample_weight=sample_weight,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        self.models['lgb'] = lgb_model
        lgb_ic = np.corrcoef(y_val, lgb_model.predict(X_val))[0, 1]
        print(f"   LightGBM Val IC: {lgb_ic:.4f}")

        # XGBoost
        try:
            import xgboost as xgb
            print("[Ensemble] Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.01,
                max_depth=4, reg_alpha=10, reg_lambda=10,
                subsample=0.5, colsample_bytree=0.4,
                objective='reg:absoluteerror',
                random_state=42, n_jobs=-1, verbosity=0,
            )
            xgb_model.fit(X_train, y_train, sample_weight=sample_weight,
                          eval_set=[(X_val, y_val)], verbose=False)
            self.models['xgb'] = xgb_model
            xgb_ic = np.corrcoef(y_val, xgb_model.predict(X_val))[0, 1]
            print(f"   XGBoost Val IC: {xgb_ic:.4f}")
        except ImportError:
            print("   XGBoost 未安装，跳过")

        # CatBoost
        try:
            from catboost import CatBoostRegressor
            print("[Ensemble] Training CatBoost...")
            cb_model = CatBoostRegressor(
                iterations=200, learning_rate=0.01,
                depth=4, l2_leaf_reg=10,
                random_seed=42, verbose=False, allow_writing_files=False,
            )
            cb_model.fit(X_train, y_train, sample_weight=sample_weight,
                         eval_set=(X_val, y_val), silent=True)
            self.models['cb'] = cb_model
            cb_ic = np.corrcoef(y_val, cb_model.predict(X_val))[0, 1]
            print(f"   CatBoost Val IC: {cb_ic:.4f}")
        except ImportError:
            print("   CatBoost 未安装，跳过")

    def predict(self, X):
        """简单平均集成"""
        preds = []
        if 'lgb' in self.models:
            preds.append(self.models['lgb'].predict(X))
        if 'xgb' in self.models:
            preds.append(self.models['xgb'].predict(X))
        if 'cb' in self.models:
            preds.append(self.models['cb'].predict(X))

        if not preds:
            raise RuntimeError("No models trained")

        return np.mean(preds, axis=0)

    def predict_weighted(self, X, weights=None):
        """加权集成 (基于验证集IC)"""
        if weights is None:
            weights = {k: 1.0 for k in self.models}

        total_w = sum(weights.values())
        pred = np.zeros(X.shape[0])
        for name, w in weights.items():
            if name in self.models:
                pred += (w / total_w) * self.models[name].predict(X)
        return pred


# ================================================================
# Part 3: Multi-Horizon Stacked Prediction
# ================================================================

class MultiHorizonPredictor:
    """多时域堆叠预测 (30min + 60min + 120min)
    借鉴: Freqtrade多时间框架, Alphalens多周期IC衰减分析
    """

    HORIZONS = [30, 60, 120]  # 分钟

    def __init__(self):
        self.models = {}  # {horizon: model}
        self.scalers = {}
        self.features = None

    def prepare_targets(self, df):
        """为每个时域创建目标变量"""
        df = df.copy()
        df['next_close'] = df['close'].shift(-1)
        for h in self.HORIZONS:
            df['future_close'] = df['close'].shift(-(h + 1))
            df[f'Target_{h}min'] = (df['future_close'] / df['next_close'] - 1) * 100
        return df

    def train(self, df_factors):
        """训练多时域模型"""
        df = self.prepare_targets(df_factors)

        # 特征选择
        exclude = {'date', 'trade_dt', 'ticker', 'close', 'open', 'high', 'low',
                   'volume', 'money', 'oi', 'time', 'next_close', 'future_close'}
        target_cols = {f'Target_{h}min' for h in self.HORIZONS}
        self.features = [c for c in df.columns if c not in exclude
                        and not c.startswith('Target_')]

        for h in self.HORIZONS:
            target = f'Target_{h}min'
            if target not in df.columns:
                continue

            df_h = df.dropna(subset=[target] + self.features).copy()
            if len(df_h) < 1000:
                continue

            for col in self.features:
                df_h[col] = df_h[col].fillna(0)

            n = len(df_h)
            train_end = int(n * 0.7)

            X_train = df_h[self.features].iloc[:train_end].values
            y_train = df_h[target].iloc[:train_end].values
            X_val = df_h[self.features].iloc[train_end:].values
            y_val = df_h[target].iloc[train_end:].values

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.005,
                num_leaves=8, max_depth=3,
                lambda_l1=15.0, lambda_l2=15.0,
                feature_fraction=0.3, bagging_fraction=0.4,
                bagging_freq=5, min_child_samples=350,
                min_split_gain=0.01, objective='regression_l1',
                random_state=42, n_jobs=-1, verbose=-1,
            )

            model.fit(X_train_s, y_train,
                      eval_set=[(X_val_s, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

            y_pred = model.predict(X_val_s)
            ic = np.corrcoef(y_val, y_pred)[0, 1]

            self.models[h] = model
            self.scalers[h] = scaler
            print(f"  [{h}min] Model trained, Val IC={ic:.4f}")

    def predict(self, df_factors):
        """生成多时域预测"""
        df = df_factors.copy()
        for col in self.features:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        X = df[self.features].values
        predictions = {}

        for h in self.HORIZONS:
            if h in self.models and h in self.scalers:
                X_s = self.scalers[h].transform(X)
                predictions[f'Pred_{h}min'] = self.models[h].predict(X_s)

        return predictions

    def consensus_signal(self, predictions, method='voting'):
        """多时域共识信号
        - voting: 多数投票
        - weighted: 按IC加权
        - cascade: 短时域确认后执行
        """
        signals = {}
        for h in self.HORIZONS:
            key = f'Pred_{h}min'
            if key in predictions:
                signals[h] = np.sign(predictions[key])

        if method == 'voting':
            # 简单多数
            stacked = np.column_stack([signals[h] for h in signals])
            consensus = np.sign(stacked.sum(axis=1))
        elif method == 'cascade':
            # 级联: 30min信号确认后才用60min, 两者一致才用120min
            consensus = np.zeros(len(list(signals.values())[0]))
            if 30 in signals:
                consensus = signals[30]  # base on shortest
                if 60 in signals:
                    consensus = np.where(consensus == signals[60], consensus, 0)
        else:
            consensus = np.zeros(len(list(signals.values())[0]))

        return consensus


# ================================================================
# Main: Run all optimizations
# ================================================================

def run_all_optimizations(base_dir, n_optuna_trials=30):
    """运行所有优化: Optuna + Ensemble + MultiHorizon"""
    factor_path = os.path.join(base_dir, "outputs", "df_factors.pkl")
    if not os.path.exists(factor_path):
        print(f"[ERROR] 因子文件不存在: {factor_path}")
        return

    df = pd.read_pickle(factor_path)
    print(f"加载因子: {df.shape}")

    results = {}

    # 1. Optuna超参优化
    print("\n" + "="*60)
    print("  PHASE 1: Optuna 贝叶斯超参优化")
    print("="*60)
    optuna_result = run_optuna_optimization(df, n_trials=n_optuna_trials)
    results['optuna'] = optuna_result

    # 2. 多时域预测
    print("\n" + "="*60)
    print("  PHASE 2: 多时域堆叠预测 (30/60/120min)")
    print("="*60)
    mh = MultiHorizonPredictor()
    mh.train(df)
    results['multi_horizon'] = {'horizons': [30, 60, 120], 'models_trained': len(mh.models)}

    # 保存
    output_path = os.path.join(base_dir, "outputs", "optimization_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n优化结果已保存: {output_path}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optuna优化 + 多模型集成')
    parser.add_argument('--trials', type=int, default=30, help='Optuna trials数')
    parser.add_argument('--mode', choices=['optuna', 'ensemble', 'multihorizon', 'all'],
                        default='all', help='优化模式')
    args = parser.parse_args()

    df = pd.read_pickle(os.path.join(BASE_DIR, "outputs", "df_factors.pkl"))

    if args.mode in ('optuna', 'all'):
        run_optuna_optimization(df, n_trials=args.trials)
    if args.mode in ('multihorizon', 'all'):
        mh = MultiHorizonPredictor()
        mh.train(df)
    if args.mode in ('all',):
        run_all_optimizations(BASE_DIR, n_optuna_trials=args.trials)
