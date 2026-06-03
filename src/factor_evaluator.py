# -*- coding: utf-8 -*-
# factor_evaluator.py — Alphalens风格因子评估模块
# 借鉴 Quantopian Alphalens (⭐4,293) + GP-Alpha-Miner + Huatai MultiFactor
#
# 核心功能:
#   1. IC 衰减曲线 — 因子预测能力的衰减速度
#   2. 分位数收益 — 因子分组的多空收益
#   3. 因子换手率 — 因子值的稳定性
#   4. 因子相关性矩阵 — 冗余检测
#   5. 综合评分 — 单因子质量排名

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


BASE_DIR = r"D:\桌面\F_Agent"


class FactorEvaluator:
    """Alphalens-style factor evaluation"""

    def __init__(self, df_factors, target_col='Target_Ret', date_col='date'):
        self.df = df_factors.copy()
        self.target_col = target_col
        self.date_col = date_col

        if self.date_col in self.df.columns:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
            self.df['_trade_date'] = self.df[self.date_col].dt.date

        # 排除非因子列
        self._exclude_cols = {
            'date', '_trade_date', 'trade_dt', 'ticker', 'close', 'open', 'high', 'low',
            'volume', 'money', 'oi', 'time', 'Hour', 'Minute', 'Minute_of_Day',
            'next_close', 'future_close', 'Target_Ret', 'Pred_Ret', 'Pred_Smooth',
            'Market_Regime', 'Should_Trade', 'Position', 'Asset_Ret',
            'Trades', 'Cost', 'Net_Ret', 'Strategy_Ret', 'Cum_Ret',
            'Signal', 'Raw_Signal', 'Confirmed_Signal', 'Pred_Rank',
            'Model_Used', 'Trade_Weight', 'Upper_Q', 'Lower_Q',
            'Leverage', 'Target_Vol', 'Position_Weight',
        }

    @property
    def factor_cols(self):
        return [c for c in self.df.columns if c not in self._exclude_cols
                and not c.startswith('_')]

    # ================================================================
    # 1. IC 分析 (Information Coefficient)
    # ================================================================

    def compute_ic(self, factor_name, periods=[1, 5, 10, 20]):
        """计算单个因子的 IC 序列和多周期 IC 衰减"""
        df = self.df.copy()
        series = df[factor_name].shift(1)  # 避免未来信息

        results = {'factor': factor_name, 'ic_decay': {}}

        # Rank IC (更稳健)
        for p in periods:
            fwd_ret = df[self.target_col].rolling(p).sum().shift(-p)
            mask = series.notna() & fwd_ret.notna()
            if mask.sum() < 100:
                results['ic_decay'][f'{p}bar'] = None
                continue
            ic = stats.spearmanr(series[mask], fwd_ret[mask])[0]
            results['ic_decay'][f'{p}bar'] = round(ic, 4)

        # 滚动 IC 稳定性
        if '_trade_date' in df.columns:
            daily_ic = []
            for dt, grp in df.groupby('_trade_date'):
                grp = grp.dropna(subset=[factor_name, self.target_col])
                if len(grp) < 20:
                    continue
                fwd = grp[self.target_col].shift(-1)
                s = grp[factor_name].shift(1)
                mask = s.notna() & fwd.notna()
                if mask.sum() < 10:
                    continue
                ic = stats.spearmanr(s[mask].astype(float), fwd[mask].astype(float))[0]
            if np.isnan(ic):
                ic = 0.0
                daily_ic.append({'date': dt, 'ic': ic})

            if daily_ic:
                df_ic = pd.DataFrame(daily_ic)
                results['daily_ic_mean'] = round(df_ic['ic'].mean(), 4)
                results['daily_ic_std'] = round(df_ic['ic'].std(), 4)
                results['ic_ir'] = round(
                    df_ic['ic'].mean() / (df_ic['ic'].std() + 1e-9), 4
                )  # Information Ratio of IC
                results['ic_positive_ratio'] = round(
                    (df_ic['ic'] > 0).mean(), 4
                )

        return results

    def rank_factors_by_ic(self, top_n=30):
        """对所有因子按IC排序"""
        print(f"\n[FactorEval] 评估 {len(self.factor_cols)} 个因子...")
        results = []
        for col in self.factor_cols:
            try:
                ic_info = self.compute_ic(col)
                ic_1bar = ic_info['ic_decay'].get('1bar', 0) or 0
                ic_ir = ic_info.get('ic_ir', 0) or 0
                results.append({
                    'factor': col,
                    'ic_1bar': ic_1bar,
                    'ic_5bar': ic_info['ic_decay'].get('5bar', 0) or 0,
                    'ic_10bar': ic_info['ic_decay'].get('10bar', 0) or 0,
                    'ic_20bar': ic_info['ic_decay'].get('20bar', 0) or 0,
                    'ic_ir': ic_ir,
                    'ic_pos_ratio': ic_info.get('ic_positive_ratio', 0) or 0,
                })
            except Exception:
                continue

        df_result = pd.DataFrame(results).sort_values('ic_1bar', key=abs, ascending=False)
        return df_result.head(top_n)

    # ================================================================
    # 2. 分位数收益分析 (Quantile Returns)
    # ================================================================

    def quantile_returns(self, factor_name, n_quantiles=5, horizon_bars=30):
        """计算因子的分位数收益 (多空组合)"""
        df = self.df.copy()
        factor = df[factor_name].shift(1)
        fwd_ret = df[self.target_col].rolling(horizon_bars).sum().shift(-horizon_bars)

        mask = factor.notna() & fwd_ret.notna()
        if mask.sum() < n_quantiles * 20:
            return None

        # 按因子值分5组
        df['_q'] = pd.qcut(factor[mask], n_quantiles, labels=False, duplicates='drop')
        df['_ret'] = fwd_ret

        quantile_rets = df.groupby('_q')['_ret'].mean()

        # 多空收益 (top - bottom)
        if len(quantile_rets) >= 2:
            long_short = quantile_rets.iloc[-1] - quantile_rets.iloc[0]
        else:
            long_short = 0

        return {
            'quantile_returns': quantile_rets.to_dict(),
            'long_short': round(long_short, 6),
            'horizon_bars': horizon_bars,
        }

    # ================================================================
    # 3. 因子换手率 (Turnover)
    # ================================================================

    def factor_turnover(self, factor_name, lookback_bars=240):
        """计算因子值的换手率 (稳定性指标)"""
        df = self.df.copy()
        series = df[factor_name].dropna()
        if len(series) < lookback_bars * 2:
            return None

        # Autocorrelation at various lags
        ac_1 = series.autocorr(lag=1)
        ac_5 = series.autocorr(lag=5)
        ac_20 = series.autocorr(lag=20)

        # Rank turnover: 今日排名 vs N日前排名的相关性
        ranks = series.rolling(lookback_bars).apply(
            lambda x: stats.rankdata(x)[-1] / len(x), raw=True
        )
        rank_corr = ranks.autocorr(lag=1) if len(ranks.dropna()) > 50 else 0

        return {
            'autocorr_1': round(ac_1, 4) if not np.isnan(ac_1) else 0,
            'autocorr_5': round(ac_5, 4) if not np.isnan(ac_5) else 0,
            'autocorr_20': round(ac_20, 4) if not np.isnan(ac_20) else 0,
            'rank_stability': round(rank_corr, 4) if not np.isnan(rank_corr) else 0,
        }

    # ================================================================
    # 4. 因子相关性矩阵 (借鉴 MultiFactor)
    # ================================================================

    def correlation_matrix(self, top_n=20):
        """计算Top因子之间的相关性矩阵"""
        ic_ranked = self.rank_factors_by_ic(top_n=top_n)
        top_factors = ic_ranked['factor'].tolist()

        df_corr = self.df[top_factors].dropna()
        if len(df_corr) < 100:
            return None, top_factors

        corr_matrix = df_corr.corr(method='spearman')

        # 识别高度相关的因子对 (>0.7)
        redundant_pairs = []
        for i in range(len(top_factors)):
            for j in range(i + 1, len(top_factors)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    redundant_pairs.append({
                        'factor_a': top_factors[i],
                        'factor_b': top_factors[j],
                        'correlation': round(corr_val, 3),
                    })

        return corr_matrix, top_factors, redundant_pairs

    # ================================================================
    # 5. 因子综合评分
    # ================================================================

    def composite_score(self, factor_name):
        """综合评分：IC * IR * 稳定性"""
        ic_info = self.compute_ic(factor_name)
        turnover = self.factor_turnover(factor_name)

        ic_1bar = abs(ic_info['ic_decay'].get('1bar', 0) or 0)
        ic_ir = ic_info.get('ic_ir', 0) or 0
        stability = turnover['rank_stability'] if turnover else 0.5

        # 综合分 = IC * IR * stability
        score = ic_1bar * ic_ir * stability

        return {
            'factor': factor_name,
            'ic_1bar': ic_1bar,
            'ic_ir': ic_ir,
            'stability': round(stability, 4),
            'composite_score': round(score, 6),
        }

    # ================================================================
    # 6. 生成完整报告
    # ================================================================

    def generate_report(self, output_path=None):
        """生成Alphalens风格的完整因子评估报告"""
        print(f"\n{'='*60}")
        print(f"  Alphalens风格因子评估报告")
        print(f"  {len(self.factor_cols)} 个因子 | {len(self.df)} 行数据")
        print(f"{'='*60}")

        # 1. IC排名
        ic_ranked = self.rank_factors_by_ic(top_n=25)
        print(f"\n--- Top 15 因子 (按|IC|排序) ---")
        for _, row in ic_ranked.head(15).iterrows():
            print(f"  {row['factor']:30s} IC={row['ic_1bar']:+.4f} "
                  f"IR={row['ic_ir']:.4f} Pos={row['ic_pos_ratio']:.2%}")

        # 2. 分位数分析 (Top 5)
        print(f"\n--- 分位数多空收益 (Top 5因子, 30bar horizon) ---")
        for _, row in ic_ranked.head(5).iterrows():
            qr = self.quantile_returns(row['factor'], n_quantiles=5, horizon_bars=30)
            if qr:
                ls = qr['long_short']
                print(f"  {row['factor']:30s} Long-Short={ls:+.4f}%")

        # 3. 换手率分析
        print(f"\n--- 因子稳定性 (Top 10) ---")
        for _, row in ic_ranked.head(10).iterrows():
            turnover = self.factor_turnover(row['factor'], lookback_bars=240)
            if turnover:
                print(f"  {row['factor']:30s} AC1={turnover['autocorr_1']:.3f} "
                      f"RankStab={turnover['rank_stability']:.3f}")

        # 4. 相关性
        corr_matrix, top_factors, redundant = self.correlation_matrix(top_n=20)
        if redundant:
            print(f"\n--- 冗余因子对 (|corr|>0.7, 共{len(redundant)}对) ---")
            for rp in redundant[:10]:
                print(f"  {rp['factor_a']:30s} ↔ {rp['factor_b']:30s} r={rp['correlation']:.3f}")

        # 5. 综合评分
        print(f"\n--- 综合评分 Top 10 ---")
        scores = []
        for col in self.factor_cols[:50]:  # 仅前50个
            try:
                score = self.composite_score(col)
                scores.append(score)
            except Exception:
                pass
        scores_sorted = sorted(scores, key=lambda x: x['composite_score'], reverse=True)[:10]
        for s in scores_sorted:
            print(f"  {s['factor']:30s} Score={s['composite_score']:.6f} "
                  f"IC={s['ic_1bar']:.4f} IR={s['ic_ir']:.4f}")

        # 保存
        if output_path is None:
            output_path = os.path.join(BASE_DIR, "outputs",
                                       f"factor_report_{datetime.now().strftime('%Y%m%d')}.csv")
        ic_ranked.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n报告已保存: {output_path}")

        return ic_ranked


def main():
    parser = argparse.ArgumentParser(description='Alphalens风格因子评估')
    parser.add_argument('--factor', type=str, default=None, help='评估单个因子')
    parser.add_argument('--top', type=int, default=25, help='排名Top N')
    args = parser.parse_args()

    factor_path = os.path.join(BASE_DIR, "outputs", "df_factors.pkl")
    if not os.path.exists(factor_path):
        print(f"[ERROR] 因子文件不存在: {factor_path}")
        sys.exit(1)

    df = pd.read_pickle(factor_path)

    # 需要 target
    if 'Target_Ret' not in df.columns:
        # 简单构造: 30min forward return
        df['next_close'] = df['close'].shift(-1)
        df['future_close'] = df['close'].shift(-31)
        df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100

    evaluator = FactorEvaluator(df)

    if args.factor:
        ic = evaluator.compute_ic(args.factor)
        qr = evaluator.quantile_returns(args.factor)
        turnover = evaluator.factor_turnover(args.factor)
        score = evaluator.composite_score(args.factor)
        print(f"\n因子: {args.factor}")
        print(f"  IC decay: {ic['ic_decay']}")
        print(f"  IC IR: {ic.get('ic_ir', 'N/A')}")
        if qr:
            print(f"  Long-Short (30bar): {qr['long_short']:+.4f}%")
        if turnover:
            print(f"  Rank Stability: {turnover['rank_stability']:.3f}")
        print(f"  Composite Score: {score['composite_score']:.6f}")
    else:
        evaluator.generate_report()


if __name__ == '__main__':
    main()
