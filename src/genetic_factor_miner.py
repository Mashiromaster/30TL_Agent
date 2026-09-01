# -*- coding: utf-8 -*-
# genetic_factor_miner.py — 遗传规划 + 符号回归因子挖掘
#
# 使用 gplearn 的遗传规划引擎，自动发现新的非线性因子表达式。
# 核心创新：
#   1. IC-based 适应度函数 — 斯皮尔曼 Rank IC 作为选择压力
#   2. 多目标优化 — IC + 低相关性（去冗余）
#   3. 与现有因子体系无缝集成
#
# 参考文献:
#   - GP-Alpha-Miner (SSRN 2022) — GP 因子挖掘框架
#   - gplearn (Stephens 2016) — Python genetic programming
#   - Kakushadze & Yu (2016) — 101 Formulaic Alphas

import pandas as pd
import numpy as np
import os
import sys
import warnings
import json
import pickle
from datetime import datetime
from collections import defaultdict

from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function, log1, sqrt1, sig1, abs1
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = r"D:\桌面\F_Agent"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "genetic_factors")
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# 自定义时序函数
# ============================================================

def _ts_rank(x, window):
    """时间序列滚动百分位排名 — 处理全零输入"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.5)  # 默认返回 0.5（中位排名）
    w = int(window)
    if len(x) < w:
        return result
    for i in range(len(x)):
        if i < w - 1:
            continue
        window_data = x[max(0, i - w + 1):i + 1]
        if np.all(window_data == 0) or np.std(window_data) < 1e-12:
            result[i] = 0.5
            continue
        rank = stats.rankdata(window_data, nan_policy='omit')
        result[i] = rank[-1] / len(rank)
    return result

def _ts_mean(x, window):
    """滚动均值"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.0)  # 默认 0
    w = int(window)
    for i in range(len(x)):
        if i < w - 1:
            continue
        val = np.nanmean(x[max(0, i - w + 1):i + 1])
        result[i] = 0.0 if np.isnan(val) else val
    return result

def _ts_std(x, window):
    """滚动标准差"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.0)
    w = int(window)
    for i in range(len(x)):
        if i < w - 1:
            continue
        val = np.nanstd(x[max(0, i - w + 1):i + 1])
        result[i] = 0.0 if np.isnan(val) else val
    return result

def _ts_delta(x, window):
    """滚动差值"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.0)
    w = int(window)
    if len(x) < w + 1:
        return result
    for i in range(len(x)):
        if i < w:
            continue
        result[i] = x[i] - x[i - w]
    return result

def _ts_max(x, window):
    """滚动最大值"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.0)
    w = int(window)
    for i in range(len(x)):
        if i < w - 1:
            continue
        val = np.nanmax(x[max(0, i - w + 1):i + 1])
        result[i] = 0.0 if np.isnan(val) else val
    return result

def _ts_min(x, window):
    """滚动最小值"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.0)
    w = int(window)
    for i in range(len(x)):
        if i < w - 1:
            continue
        val = np.nanmin(x[max(0, i - w + 1):i + 1])
        result[i] = 0.0 if np.isnan(val) else val
    return result

def _sign(x):
    """符号函数"""
    return np.sign(x)

def _abs_np(x):
    """绝对值"""
    return np.abs(x)

def _relu(x):
    """ReLU 激活"""
    return np.maximum(0, x)


# ============================================================
# 注册函数到 gplearn function set
# 使用 gplearn 内置保护函数 (sqrt1, log1 自带 NaN/Inf 保护)
# ============================================================

sign_func  = make_function(function=_sign,   name='sign', arity=1)
relu_func  = make_function(function=_relu,   name='relu', arity=1)

ts_rank_5   = make_function(function=lambda x: _ts_rank(x, 5),   name='ts_rank_5',  arity=1)
ts_rank_20  = make_function(function=lambda x: _ts_rank(x, 20),  name='ts_rank_20', arity=1)
ts_mean_5   = make_function(function=lambda x: _ts_mean(x, 5),   name='ts_mean_5',  arity=1)
ts_mean_20  = make_function(function=lambda x: _ts_mean(x, 20),  name='ts_mean_20', arity=1)
ts_std_5    = make_function(function=lambda x: _ts_std(x, 5),    name='ts_std_5',   arity=1)
ts_std_20   = make_function(function=lambda x: _ts_std(x, 20),   name='ts_std_20',  arity=1)
ts_delta_5  = make_function(function=lambda x: _ts_delta(x, 5),  name='ts_delta_5', arity=1)
ts_delta_20 = make_function(function=lambda x: _ts_delta(x, 20), name='ts_delta_20',arity=1)
ts_max_20  = make_function(function=lambda x: _ts_max(x, 20),   name='ts_max_20',  arity=1)
ts_min_20  = make_function(function=lambda x: _ts_min(x, 20),   name='ts_min_20',  arity=1)

CUSTOM_FUNCTIONS = [
    sqrt1, log1, sig1, abs1, sign_func, relu_func,
    ts_rank_5, ts_rank_20, ts_mean_5, ts_mean_20,
    ts_std_5, ts_std_20, ts_delta_5, ts_delta_20,
    ts_max_20, ts_min_20,
]


# ============================================================
# IC-based 适应度函数
# ============================================================

def _ic_fitness(y_true, y_pred, w=None):
    """
    斯皮尔曼 Rank IC 作为适应度。

    在量化因子挖掘中，MSE 是错误的目标：
    - MSE 惩罚预测值的大小偏差 → 优化出接近常数的"安全"因子
    - IC 只关心排序方向 → 真正选出有预测能力的因子

    返回 [IC_signed, IC_abs] 两个适应度分数（gplearn 支持多目标）
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 30:
        return [-1e-6, 1e-6]

    yt = y_true[mask]
    yp = y_pred[mask]

    yt_clip = np.clip(yt, np.percentile(yt, 1), np.percentile(yt, 99))
    yp_clip = np.clip(yp, np.percentile(yp, 1), np.percentile(yp, 99))

    if np.std(yp_clip) < 1e-12:
        return [-1e-6, 1e-6]

    ic, _ = stats.spearmanr(yt_clip, yp_clip)
    if np.isnan(ic):
        return [-1e-6, 1e-6]

    return [ic, abs(ic)]


# ============================================================
# 因子去重与后处理
# ============================================================

def _deduplicate_factors(df_factors, threshold=0.92):
    """基于相关系数去重"""
    factor_cols = [c for c in df_factors.columns if c.startswith('GP_')]
    if len(factor_cols) <= 1:
        return df_factors, []

    corr_matrix = df_factors[factor_cols].corr().abs()
    removed = []
    kept = set(factor_cols)

    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            fi, fj = factor_cols[i], factor_cols[j]
            if fi not in kept or fj not in kept:
                continue
            if corr_matrix.loc[fi, fj] >= threshold:
                loser = fj if len(fj) > len(fi) else fi
                if loser in kept:
                    kept.discard(loser)
                    removed.append(loser)

    drop_cols = [c for c in factor_cols if c not in kept]
    if drop_cols:
        print(f"[GP Mine] 去重: 移除 {len(drop_cols)} 个高相关因子")
    return df_factors.drop(columns=drop_cols, errors='ignore'), removed


def _rank_factors_by_ic(df_factors, target_col='Target_Ret'):
    """按 IC 绝对值排序因子"""
    factor_cols = [c for c in df_factors.columns if c.startswith('GP_')]
    if not factor_cols:
        return []

    mask = df_factors[target_col].notna()
    rankings = []
    for col in factor_cols:
        valid = mask & df_factors[col].notna()
        if valid.sum() < 50:
            continue
        ic, _ = stats.spearmanr(
            df_factors.loc[valid, target_col],
            df_factors.loc[valid, col]
        )
        rankings.append((col, ic, abs(ic)))

    rankings.sort(key=lambda x: -x[2])
    return rankings


def _simplify_program_name(prog_str, feature_names):
    """简化表达式名"""
    max_len = 30
    if len(prog_str) <= max_len:
        return prog_str
    short = prog_str.replace('(', '').replace(')', '').replace(' ', '_')
    return short[:max_len]


# ============================================================
# 核心挖掘类
# ============================================================

class GeneticFactorMiner:
    """遗传规划因子挖掘器"""

    def __init__(
        self,
        population_size=2000,
        generations=25,
        tournament_size=20,
        parsimony_coefficient=0.001,
        max_samples=0.5,
        random_state=42,
        n_jobs=-1,
        init_depth=(2, 5),
        const_range=(-5.0, 5.0),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.parsimony_coefficient = parsimony_coefficient
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.init_depth = init_depth
        self.const_range = const_range
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation

        self._transformer = None
        self._best_programs = []

    def fit(self, X, y, feature_names=None):
        """运行遗传规划因子挖掘"""
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or list(X.columns)
            X_arr = X.values
        else:
            X_arr = X
            feature_names = feature_names or [f'F_{i}' for i in range(X_arr.shape[1])]

        if isinstance(y, pd.Series):
            y_arr = y.values
        else:
            y_arr = y

        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)

        print(f"[GP Mine] 开始遗传规划挖掘...")
        print(f"  输入: {X_arr.shape[1]} 特征, {X_arr.shape[0]} 样本")
        print(f"  种群: {self.population_size}, 代数: {self.generations}")

        self._transformer = SymbolicTransformer(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            function_set=CUSTOM_FUNCTIONS,
            parsimony_coefficient=self.parsimony_coefficient,
            max_samples=self.max_samples,
            init_depth=self.init_depth,
            const_range=self.const_range,
            p_crossover=self.p_crossover,
            p_subtree_mutation=self.p_subtree_mutation,
            p_hoist_mutation=self.p_hoist_mutation,
            p_point_mutation=self.p_point_mutation,
            metric=_ic_fitness,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1,
            feature_names=feature_names,
            stopping_criteria=0.01,
        )

        self._transformer.fit(X_arr, y_arr)
        self._best_programs = self._transformer._best_programs

        print(f"[GP Mine] 挖掘完成 — 发现 {len(self._best_programs)} 个候选因子")
        return self

    def transform(self, X):
        """使用挖掘出的因子程序转换数据"""
        if self._transformer is None:
            raise RuntimeError("请先调用 fit()")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)

        result = self._transformer.transform(X_arr)

        cols = []
        for i, program in enumerate(self._best_programs):
            prog_str = str(program)
            short_name = _simplify_program_name(prog_str, [])
            col_name = f"GP_{i:03d}_{short_name}"
            cols.append(col_name[:60])

        df = pd.DataFrame(result, columns=cols)
        return df

    def fit_transform(self, X, y, feature_names=None):
        """fit + transform"""
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_factor_programs(self):
        """返回发现的因子程序列表"""
        return [
            {
                'index': i,
                'program': str(p),
                'length': len(str(p)),
                'fitness': float(p.fitness_) if hasattr(p, 'fitness_') else None,
            }
            for i, p in enumerate(self._best_programs)
        ]

    def save(self, path=None):
        """持久化挖掘结果"""
        if path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(CACHE_DIR, f"genetic_programs_{ts}.pkl")

        data = {
            'best_programs': [str(p) for p in self._best_programs],
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'random_state': self.random_state,
            },
            'timestamp': datetime.now().isoformat(),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[GP Mine] 已保存到: {path}")
        return path


# ============================================================
# 一键挖掘封装
# ============================================================

def mine_genetic_factors(
    df_factors,
    target_col='Target_Ret',
    exclude_patterns=None,
    population_size=1500,
    generations=20,
    max_new_factors=30,
    min_ic_threshold=0.01,
):
    """一键遗传因子挖掘"""

    if exclude_patterns is None:
        exclude_patterns = [
            'date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
            'money', 'oi', 'Target_Ret', 'Pred_Ret', 'Market_Regime',
            'Hour', 'Minute', 'Minute_of_Day', 'Trades',
        ]

    print("\n" + "=" * 60)
    print("  遗传规划因子挖掘 (Genetic Programming Factor Mining)")
    print("=" * 60)

    # 1. 确定输入特征
    feature_cols = []
    for c in df_factors.columns:
        skip = False
        for pat in exclude_patterns:
            if pat in c or c.startswith('_') or c.startswith('GP_'):
                skip = True
                break
        if not skip:
            feature_cols.append(c)

    print(f"[GP Mine] 输入特征: {len(feature_cols)} 个")

    # 2. 准备数据
    df_clean = df_factors.dropna(subset=[target_col]).copy()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    print(f"[GP Mine] 有效样本: {len(df_clean)} 行")

    if len(df_clean) < 500:
        print("[GP Mine ERROR] 样本量不足")
        return df_factors, {'error': 'insufficient_samples', 'n_samples': len(df_clean)}

    # 3. 运行挖掘
    miner = GeneticFactorMiner(
        population_size=population_size,
        generations=generations,
        random_state=42,
    )
    df_new = miner.fit_transform(X, y, feature_names=feature_cols)

    # 4. 合并
    df_result = df_clean.copy()
    for col in df_new.columns:
        df_result[col] = df_new[col].values

    # 5. 去重
    df_result, removed = _deduplicate_factors(df_result)

    # 6. IC 排名
    rankings = _rank_factors_by_ic(df_result, target_col)
    print(f"\n[GP Mine] 因子排名 (按 |IC|):")
    for name, ic, abs_ic in rankings[:20]:
        flag = " ✓" if abs_ic >= min_ic_threshold else ""
        print(f"  {name}: IC={ic:+.4f}  |IC|={abs_ic:.4f}{flag}")

    # 7. 过滤
    good_factors = [name for name, _, abs_ic in rankings if abs_ic >= min_ic_threshold]
    good_factors = good_factors[:max_new_factors]
    bad_factors = [c for c in df_result.columns if c.startswith('GP_') and c not in good_factors]

    if bad_factors:
        df_result = df_result.drop(columns=bad_factors, errors='ignore')

    # 8. 报告
    report = {
        'n_input_features': len(feature_cols),
        'n_samples': len(df_clean),
        'n_candidates': len(df_new.columns),
        'n_after_dedup': len(df_new.columns) - len(removed),
        'n_final': len(good_factors),
        'final_factors': good_factors,
        'rankings': rankings[:30],
        'programs': miner.get_factor_programs()[:30],
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n[GP Mine] 最终保留: {len(good_factors)} 个因子")
    for f in good_factors:
        print(f"  → {f}")

    # 9. 对齐回原始 df_factors
    final_new_cols = [c for c in df_result.columns if c.startswith('GP_')]
    if final_new_cols:
        df_out = df_factors.copy()
        for col in final_new_cols:
            df_out[col] = np.nan
            common = df_factors.index.intersection(df_result.index)
            if len(common) > 0:
                df_out.loc[common, col] = df_result.loc[common, col]
            df_out[col] = df_out[col].ffill().fillna(0)
        print(f"[GP Mine] 已将 {len(final_new_cols)} 个新因子合并")
    else:
        df_out = df_factors
        print("[GP Mine WARNING] 没有通过 IC 门槛的因子")

    print("=" * 60 + "\n")
    return df_out, report


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Genetic Factor Mining')
    parser.add_argument('--data', type=str, help='Path to factor dataframe pickle')
    parser.add_argument('--population', type=int, default=1500)
    parser.add_argument('--generations', type=int, default=20)
    parser.add_argument('--min-ic', type=float, default=0.01)
    parser.add_argument('--max-factors', type=int, default=30)
    parser.add_argument('--save', action='store_true', default=True)
    args = parser.parse_args()

    if args.data and os.path.exists(args.data):
        df = pd.read_pickle(args.data)
    else:
        default_path = os.path.join(OUTPUT_DIR, "factor_df.pkl")
        if os.path.exists(default_path):
            df = pd.read_pickle(default_path)
        else:
            print("[GP Mine ERROR] 请指定 --data")
            sys.exit(1)

    print(f"[GP Mine] 数据: {len(df)} 行, {len(df.columns)} 列")
    if 'Target_Ret' not in df.columns:
        print("[GP Mine ERROR] 缺少 Target_Ret")
        sys.exit(1)

    df_new, report = mine_genetic_factors(
        df, target_col='Target_Ret',
        population_size=args.population,
        generations=args.generations,
        min_ic_threshold=args.min_ic,
        max_new_factors=args.max_factors,
    )

    if args.save:
        output_path = os.path.join(OUTPUT_DIR, "genetic_factors_result.pkl")
        df_new.to_pickle(output_path)
        print(f"[GP Mine] 已保存: {output_path}")

        report_path = os.path.join(CACHE_DIR, "genetic_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"[GP Mine] 报告已保存: {report_path}")


if __name__ == '__main__':
    main()
