# -*- coding: utf-8 -*-
# moe_model.py — 双层 LightGBM Mixture-of-Experts (V1.0)
#
# 架构 (spec: docs/superpowers/specs/2026-09-01-lightgbm-moe-design.md):
#   第一层 Gating Layer : LightGBM 多分类门控 → softmax(g0,g1,g2)
#   第二层 Expert Layer : 3 个 regime 专家 (正常/高波动/趋势)，各自只在对应 regime 训练
#   软路由融合          : y_hat = g0*f0(x) + g1*f1(x) + g2*f2(x)
#
# 上线判据 (CLAUDE.md rule #5): MoE 组合 IC >= 现有 base/active 硬路由基线 IC 才切换默认路径。

import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import RobustScaler

import LightGBM_model as base_lgb

REGIME_NAMES = {0: '正常', 1: '高波动', 2: '趋势'}
MIN_REGIME_SAMPLES = 1000  # 某 regime 训练样本不足则回退全样本 base 专家

# 门控层特征子集：只用于判断市场状态，不重学预测
GATE_FEATURE_CANDIDATES = [
    'RV_Percentile', 'Vol_Regime', 'ATR_14', 'Vol_Surge', 'RV_30', 'RV_120',
    'Trend_Consistency', 'Is_High_Vol', 'CN_US_10Y_Spread', 'Basis_ZScore_20',
]
GATE_FEATURE_PATTERNS = ['Liquidity', 'Repo_', 'SHIBOR_']  # 流动性/资金面类，存在则纳入


def _expert_params(model_type):
    """复用 LightGBM_model 已验证的强正则超参。"""
    if model_type == 'highvol':
        return dict(n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=4,
                    lambda_l1=10.0, lambda_l2=10.0, feature_fraction=0.4,
                    bagging_fraction=0.5, bagging_freq=5, min_child_samples=250,
                    min_split_gain=0.01, objective='regression_l1',
                    random_state=42, n_jobs=-1, verbose=-1)
    return dict(n_estimators=200, learning_rate=0.005, num_leaves=8, max_depth=3,
                lambda_l1=15.0, lambda_l2=15.0, feature_fraction=0.3,
                bagging_fraction=0.4, bagging_freq=5, min_child_samples=350,
                min_split_gain=0.01, objective='regression_l1',
                random_state=42, n_jobs=-1, verbose=-1)


def _fit_regressor(params, X_tr, y_tr, X_val, y_val, sw=None):
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr, sample_weight=sw,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                         lgb.log_evaluation(period=0)])
    return model


def train_experts(X_train, y_train, regime_train, X_val, y_val, sw_train=None):
    """训练 3 个 regime 专家。样本不足的 regime 回退为全样本 base 专家。

    返回 (experts: dict{0,1,2 -> model}, fallback: set{回退的 regime_id})。
    """
    experts, fallback = {}, set()

    # 全样本 base 专家：既是 regime 0 的专家，也是不足 regime 的回退模型
    base_expert = _fit_regressor(_expert_params('base'),
                                 X_train, y_train, X_val, y_val, sw_train)

    for rid in (0, 1, 2):
        mask = (regime_train == rid)
        if rid == 0:
            experts[0] = base_expert
            print(f"[MoE] Expert 0 ({REGIME_NAMES[0]}): 全样本 base ({mask.sum()} 条同 regime)")
            continue

        if mask.sum() < MIN_REGIME_SAMPLES:
            experts[rid] = base_expert
            fallback.add(rid)
            print(f"[MoE] Expert {rid} ({REGIME_NAMES[rid]}): 样本仅 {mask.sum()} < "
                  f"{MIN_REGIME_SAMPLES} → 回退全样本 base 专家")
            continue

        sw = sw_train[mask] if sw_train is not None else None
        # regime 1/2 均用高容量强正则配置；验证集用全 val 片段
        experts[rid] = _fit_regressor(_expert_params('highvol'),
                                      X_train[mask], y_train[mask], X_val, y_val, sw)
        print(f"[MoE] Expert {rid} ({REGIME_NAMES[rid]}): 独立训练 ({mask.sum()} 条)")

    return experts, fallback


def _resolve_gate_features(df):
    feats = [f for f in GATE_FEATURE_CANDIDATES if f in df.columns]
    for c in df.columns:
        if any(p in c for p in GATE_FEATURE_PATTERNS) and c not in feats:
            feats.append(c)
    return feats


def train_gate(gate_X_val, y_val, experts, X_val_full):
    """在验证集上训练门控多分类器。

    标签 = 在该验证样本上预测误差 |f_i - y| 最小的专家索引（用验证集避免标签泄漏）。
    gate_X_val : 门控特征子集 (验证集)
    X_val_full : 全特征 (验证集)，供各专家预测
    """
    preds = np.column_stack([experts[i].predict(X_val_full) for i in (0, 1, 2)])
    errors = np.abs(preds - y_val.reshape(-1, 1))
    labels = errors.argmin(axis=1)

    dist = np.bincount(labels, minlength=3)
    print(f"[MoE] 门控标签分布 (验证集最优专家): "
          f"正常={dist[0]} 高波={dist[1]} 趋势={dist[2]}")

    gate = LGBMClassifier(
        n_estimators=100, learning_rate=0.02, num_leaves=8, max_depth=3,
        lambda_l1=5.0, lambda_l2=5.0, feature_fraction=0.6,
        min_child_samples=100, objective='multiclass', num_class=3,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    gate.fit(gate_X_val, labels)
    return gate


def moe_predict(gate, experts, gate_X, X_full):
    """软路由融合预测。返回 (y_hat, gate_weights[n,3])。"""
    weights = gate.predict_proba(gate_X)  # [n, k] softmax
    # 门控类别可能不足 3（某专家从未成为最优），对齐到 3 列
    if weights.shape[1] < 3:
        full = np.zeros((weights.shape[0], 3))
        for col_idx, cls in enumerate(gate.classes_):
            full[:, int(cls)] = weights[:, col_idx]
        weights = full

    expert_preds = np.column_stack([experts[i].predict(X_full) for i in (0, 1, 2)])
    y_hat = (weights * expert_preds).sum(axis=1)
    return y_hat, weights


def _ic(y, yhat):
    if len(y) < 3 or np.std(yhat) < 1e-12:
        return np.nan
    return np.corrcoef(y, yhat)[0, 1]


def run_process(base_dir, max_lookback_months=12, time_decay_half_life=90):
    print("\n" + "=" * 50)
    print("MoE: 双层 LightGBM Mixture-of-Experts 训练")
    print("=" * 50)

    FACTOR_FILE = os.path.join(base_dir, "outputs/df_factors.pkl")
    PRED_FILE = os.path.join(base_dir, "outputs/df_predictions_moe.pkl")
    MODEL_FILE = os.path.join(base_dir, "models/moe_model.pkl")

    if not os.path.exists(FACTOR_FILE):
        print(f"[MoE ERROR] 找不到因子文件: {FACTOR_FILE}")
        return False
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    df = pd.read_pickle(FACTOR_FILE)
    df_model, features = base_lgb.prepare_training_data(df, prediction_horizon=30)
    gate_features = _resolve_gate_features(df_model)
    print(f"[MoE] 全特征: {len(features)}, 门控特征: {len(gate_features)} → {gate_features}")

    # 时间序列 70/15/15 切分（与现有一致）
    n = len(df_model)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    train_df = df_model.iloc[:train_end].copy()
    val_df = df_model.iloc[train_end:val_end].copy()
    test_df = df_model.iloc[val_end:].copy()

    # 12 月窗口截断
    if max_lookback_months:
        cutoff = test_df['date'].min() - pd.Timedelta(days=max_lookback_months * 30)
        train_df = train_df[train_df['date'] >= cutoff].copy()
        print(f"[MoE] 训练窗口: {max_lookback_months}月 "
              f"({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")

    # 90d 时间衰减权重
    sw_train = None
    if time_decay_half_life and len(train_df) > 0:
        age = (train_df['date'].max() - train_df['date']).dt.total_seconds() / 86400
        sw_train = np.exp(-np.log(2) * age / time_decay_half_life).values

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_val = scaler.transform(val_df[features])
    X_test = scaler.transform(test_df[features])
    y_train = train_df['Target_Ret'].values
    y_val = val_df['Target_Ret'].values
    y_test = test_df['Target_Ret'].values

    def _regime(d):
        return d['Market_Regime'].values if 'Market_Regime' in d.columns else np.zeros(len(d))
    regime_train, regime_test = _regime(train_df), _regime(test_df)

    print(f"[MoE] 训练/验证/测试: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    # ── 第二层: 专家 ──
    print("\n[MoE] 训练专家层...")
    experts, fallback = train_experts(X_train, y_train, regime_train,
                                      X_val, y_val, sw_train)

    # ── 第一层: 门控 ──
    print("\n[MoE] 训练门控层 (验证集打标签)...")
    gate_X_val = val_df[gate_features].fillna(0).values
    gate = train_gate(gate_X_val, y_val, experts, X_val)

    # ── 软路由测试集预测 ──
    gate_X_test = test_df[gate_features].fillna(0).values
    y_moe, w_test = moe_predict(gate, experts, gate_X_test, X_test)
    moe_ic = _ic(y_test, y_moe)

    # ── 基线: 现有 base/active 硬路由 ──
    baseline_ic = _baseline_ic(X_train, y_train, regime_train, X_val, y_val,
                               X_test, y_test, regime_test, sw_train)

    print("\n" + "=" * 50)
    print(f"[MoE] 测试集整体 IC : {moe_ic:.4f}")
    print(f"[MoE] 基线 (硬路由) IC: {baseline_ic:.4f}")
    print(f"[MoE] 提升          : {moe_ic - baseline_ic:+.4f}")
    print("=" * 50)

    print("\n[MoE] 分状态 IC:")
    for rid in (0, 1, 2):
        m = (regime_test == rid)
        if m.sum() > 10:
            print(f"  - {REGIME_NAMES[rid]}: IC={_ic(y_test[m], y_moe[m]):.4f}, n={m.sum()}")

    dominant = w_test.argmax(axis=1)
    dist = np.bincount(dominant, minlength=3) / max(len(dominant), 1)
    print("\n[MoE] 门控主导专家占比:")
    for rid in (0, 1, 2):
        flag = "  <5% 疑似坍塌" if dist[rid] < 0.05 else ""
        print(f"  - {REGIME_NAMES[rid]}: {dist[rid]:.1%}{flag}")

    # 保存产物
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'experts': experts, 'gate': gate,
            'gate_features': gate_features, 'features': features,
            'scaler': scaler, 'regime_fallback': fallback,
            'meta': {'moe_ic': float(moe_ic), 'baseline_ic': float(baseline_ic),
                     'beats_baseline': bool(moe_ic >= baseline_ic)},
        }, f)
    print(f"\n[MoE] 模型已保存: {MODEL_FILE}")

    out = test_df.copy()
    out['Pred_Ret'] = y_moe
    out['gate_w0'], out['gate_w1'], out['gate_w2'] = w_test[:, 0], w_test[:, 1], w_test[:, 2]
    out['Model_Used'] = [REGIME_NAMES[i] for i in dominant]
    cols = [c for c in ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 'Pred_Ret',
                        'oi', 'volume', 'money', 'Market_Regime',
                        'gate_w0', 'gate_w1', 'gate_w2', 'Model_Used'] if c in out.columns]
    out[cols].to_pickle(PRED_FILE)
    print(f"[MoE] 预测已保存: {PRED_FILE}")

    if moe_ic >= baseline_ic:
        print("\n[MoE] IC 不倒退，可切换默认推理路径。")
    else:
        print("\n[MoE] IC 低于基线，保留为实验，不改 inference 默认路径 (rule #5)。")

    return True


def _baseline_ic(X_train, y_train, regime_train, X_val, y_val,
                 X_test, y_test, regime_test, sw_train):
    """复现现有 base/active 硬路由，用于同 test 集对比。"""
    base = _fit_regressor(_expert_params('base'), X_train, y_train, X_val, y_val, sw_train)

    active_mask = np.isin(regime_train, [1, 2])
    active = None
    if active_mask.sum() > 1000:
        sw = sw_train[active_mask] if sw_train is not None else None
        active = _fit_regressor(_expert_params('highvol'),
                                X_train[active_mask], y_train[active_mask],
                                X_val, y_val, sw)

    y_pred = np.zeros(len(y_test))
    test_active = np.isin(regime_test, [1, 2])
    if active is not None and test_active.sum() > 0:
        y_pred[test_active] = active.predict(X_test[test_active])
    other = ~test_active
    y_pred[other] = base.predict(X_test[other])
    return _ic(y_test, y_pred)
