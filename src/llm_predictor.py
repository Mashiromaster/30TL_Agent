# -*- coding: utf-8 -*-
# llm_predictor.py — DeepSeek 直接预测 TL 国债期货方向 (V1.0)
# 实验：大模型 vs LightGBM 日频方向预测能力对比

import pandas as pd
import numpy as np
import os
import json
import time
import re
from datetime import datetime


class LLMDirectPredictor:
    """Use DeepSeek to directly predict daily TL futures direction."""

    def __init__(self, base_dir, api_key=None):
        self.base_dir = base_dir
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")

        # Load model training data (has all factors + Target_Ret)
        import LightGBM_model
        factor_path = os.path.join(base_dir, "outputs/df_factors.pkl")
        df = pd.read_pickle(factor_path)
        self.df_model, self.features = LightGBM_model.prepare_training_data(df)

        # Test period only
        n = len(self.df_model)
        val_end = int(n * 0.85)
        self.test_df = self.df_model.iloc[val_end:].copy()
        self.test_dates = sorted(self.test_df['trade_dt'].unique())

        # Build daily aggregates
        self._build_daily_data()

        # Skip first 10 days (need warmup for context building)
        self.test_dates_predictable = [dt for dt in self.test_dates
                                       if self._build_context(dt) is not None]

    # ─── Daily aggregation ─────────────────────────────────

    def _build_daily_data(self):
        """Build daily-level OHLCV + factor data from bar-level test data."""
        records = []
        for dt in self.test_dates:
            day = self.test_df[self.test_df['trade_dt'] == dt]
            if len(day) == 0:
                continue

            last = day.iloc[-1]
            record = {
                'trade_dt': dt,
                'date': last['date'],
                'open': day['close'].iloc[0],
                'close': last['close'],
                'high': day['close'].max(),
                'low': day['close'].min(),
                'volume': day.get('volume', pd.Series([0])).sum(),
                'market_regime': int(last['Market_Regime']),
                'target_ret': last['Target_Ret'],  # next bar's return (30min forward)
            }

            # Add top factor values at end of day
            for f in self.features:
                if f in day.columns:
                    record[f] = float(last[f]) if pd.notna(last[f]) else 0.0

            records.append(record)

        self.daily_df = pd.DataFrame(records)
        self.daily_df['daily_return'] = self.daily_df['close'].pct_change()

    # ─── Prompt building ────────────────────────────────────

    def _build_context(self, trade_dt):
        """Build market context prompt for a given trading day."""
        daily = self.daily_df
        mask = daily['trade_dt'] <= trade_dt
        past = daily[mask].tail(11)  # Include the target day itself, then last 10 before it

        if len(past) < 2:
            return None

        current = past.iloc[-1]  # The target day
        recent = past.iloc[:-1].tail(10)  # Previous 10 days

        # Price history
        price_lines = []
        for _, row in recent.iterrows():
            dt_str = str(row['trade_dt'])[:10]
            ret = row.get('daily_return', np.nan)
            ret_str = f"{ret*100:+.2f}%" if pd.notna(ret) else "N/A"
            price_lines.append(f"  {dt_str}: close={row['close']:.2f}, 日涨跌={ret_str}")

        regime_name = ['正常', '高波动', '趋势'][int(current['market_regime'])]

        # Key factor snapshot
        top_factors = [
            'RV_30', 'Mid_Momentum_1M', 'Basis_ZScore_20', 'RSI',
            'Short_Momentum_5D', 'MACD_Hist', 'Spread_Ratio',
            'Smart_Money', 'Cum_Imbalance_30', 'ATR_14',
        ]
        factor_lines = []
        for f in top_factors:
            if f in current.index and pd.notna(current[f]):
                factor_lines.append(f"  {f}: {current[f]:.4f}")

        # Volatility context
        vol = current.get('RV_30', 'N/A')
        atr = current.get('ATR_14', 'N/A')

        prompt = f"""你是中国国债期货(TL 30年期)专业量化交易员。你的任务是分析当前市场状况，预测下一交易日TL主力合约的涨跌方向。

## 近期行情（过去10个交易日）
{chr(10).join(price_lines)}

## 当前市场状态
- 交易日期: {str(trade_dt)[:10]}
- 今日收盘价: {current['close']:.2f}
- 今日最高: {current.get('high', 'N/A')}
- 今日最低: {current.get('low', 'N/A')}
- 市场状态: {regime_name}
- 30日波动率: {vol}
- 14日ATR: {atr}

## 关键量化因子
{chr(10).join(factor_lines)}

## 任务
基于以上市场数据，预测下一交易日TL主力合约的涨跌方向。

你必须只回复一个JSON对象（不要其他文字），格式如下：
{{"direction": "UP", "confidence": 0.65, "reasoning": "关键判断依据(中文20字以内)"}}

direction必须是 "UP"(上涨)、"DOWN"(下跌) 或 "FLAT"(持平/不确定) 之一。
confidence是0.0到1.0之间的置信度。
reasoning简要说明你的核心判断逻辑。"""

        return prompt

    # ─── Prediction ─────────────────────────────────────────

    def predict_single(self, trade_dt):
        """Predict direction for a single trading day."""
        if not self.api_key:
            return {'direction': 'FLAT', 'confidence': 0.0,
                    'reasoning': '未设置DEEPSEEK_API_KEY'}

        prompt = self._build_context(trade_dt)
        if prompt is None:
            return {'direction': 'FLAT', 'confidence': 0.0,
                    'reasoning': '历史数据不足'}

        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是专业中国国债期货交易员。只回复JSON，不要其他文字。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON from response
            result = self._parse_response(raw)
            return result
        except Exception as e:
            return {'direction': 'FLAT', 'confidence': 0.0,
                    'reasoning': f'API错误: {str(e)[:50]}'}

    def _parse_response(self, raw):
        """Extract JSON from LLM response (handle markdown code blocks)."""
        # Try direct JSON first
        try:
            data = json.loads(raw)
            return self._validate_prediction(data)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._validate_prediction(data)
            except json.JSONDecodeError:
                pass

        # Try finding any JSON object
        json_match = re.search(r'\{[^}]+\}', raw)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return self._validate_prediction(data)
            except json.JSONDecodeError:
                pass

        # Fallback: try to find direction keyword
        raw_upper = raw.upper()
        if 'UP' in raw_upper and 'DOWN' not in raw_upper:
            direction = 'UP'
        elif 'DOWN' in raw_upper and 'UP' not in raw_upper:
            direction = 'DOWN'
        else:
            direction = 'FLAT'

        return {'direction': direction, 'confidence': 0.3,
                'reasoning': f'解析失败,从原始回复推断: {raw[:80]}'}

    def _validate_prediction(self, data):
        """Validate and normalize prediction dict."""
        direction = str(data.get('direction', 'FLAT')).upper()
        if direction not in ('UP', 'DOWN', 'FLAT'):
            direction = 'FLAT'

        confidence = float(data.get('confidence', 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(data.get('reasoning', ''))[:200]

        return {'direction': direction, 'confidence': confidence,
                'reasoning': reasoning}

    # ─── Batch prediction ───────────────────────────────────

    def predict_test_period(self, start_idx=0, end_idx=None, delay=1.0):
        """Predict for the entire test period. Returns DataFrame with predictions.

        Args:
            start_idx: Start from this index (for resume)
            end_idx: End at this index (None = all)
            delay: Seconds between API calls (rate limiting)
        """
        if end_idx is None:
            end_idx = len(self.test_dates)

        dates_to_predict = self.test_dates_predictable[start_idx:end_idx]
        results = []
        n_total = len(dates_to_predict)

        print(f"[LLM Predict] 开始预测 {n_total} 个交易日...")
        print(f"[LLM Predict] 预计耗时 ~{n_total * delay / 60:.1f} 分钟")

        for i, dt in enumerate(dates_to_predict):
            dt_str = str(dt)[:10]
            result = self.predict_single(dt)
            result['trade_dt'] = dt_str
            results.append(result)

            # Get actual next-day info for immediate feedback
            daily_mask = self.daily_df['trade_dt'] == dt
            current_day = self.daily_df[daily_mask]
            if len(current_day) > 0:
                result['close_today'] = float(current_day['close'].iloc[0])
                result['market_regime'] = int(current_day['market_regime'].iloc[0])

            direction_symbol = {'UP': 'LONG', 'DOWN': 'SHORT', 'FLAT': 'FLAT'}[result['direction']]
            print(f"  [{i+1}/{n_total}] {dt_str} {direction_symbol} {result['direction']:4s} "
                  f"conf={result['confidence']:.2f} | {result['reasoning'][:40]}")

            if i < n_total - 1:
                time.sleep(delay)

        self.df_llm_pred = pd.DataFrame(results)
        self._attach_actuals()
        return self.df_llm_pred

    def _attach_actuals(self):
        """Attach next-day actual return to LLM predictions for evaluation."""
        if not hasattr(self, 'df_llm_pred') or self.df_llm_pred is None:
            return

        actuals = []
        for _, row in self.df_llm_pred.iterrows():
            dt = row['trade_dt']
            # Find next trading day's close to compute actual return
            day_idx = self.daily_df['trade_dt'].searchsorted(dt)
            current_mask = self.daily_df['trade_dt'] == dt
            current = self.daily_df[current_mask]

            if len(current) == 0 or day_idx >= len(self.daily_df) - 1:
                actuals.append({'actual_return': None, 'is_correct': None})
                continue

            # Next day's close vs today's close
            today_close = current['close'].iloc[0]
            next_day = self.daily_df.iloc[day_idx + 1]
            next_close = next_day['close']
            actual_ret = (next_close / today_close - 1) * 100

            direction = row['direction']
            if direction == 'UP':
                is_correct = actual_ret > 0
            elif direction == 'DOWN':
                is_correct = actual_ret < 0
            else:
                is_correct = None  # FLAT = no bet

            actuals.append({
                'actual_return_pct': round(float(actual_ret), 4),
                'is_correct': is_correct,
            })

        actual_df = pd.DataFrame(actuals)
        self.df_llm_pred = pd.concat([self.df_llm_pred, actual_df], axis=1)

    # ─── Daily backtest ─────────────────────────────────────

    def daily_backtest(self):
        """Simple daily P&L backtest: direction × next_day_return."""
        if not hasattr(self, 'df_llm_pred') or self.df_llm_pred is None:
            return {'error': '请先运行 predict_test_period()'}

        df = self.df_llm_pred.copy()
        df = df[df['actual_return_pct'].notna()].copy()

        # Map direction to position
        dir_to_pos = {'UP': 1, 'DOWN': -1, 'FLAT': 0}
        df['position'] = df['direction'].map(dir_to_pos)

        # Daily P&L
        df['daily_pnl'] = df['position'] * df['actual_return_pct'] / 100

        # Metrics
        n_trades = (df['position'] != 0).sum()
        n_correct = (df['is_correct'] == True).sum()
        n_evaluated = df['is_correct'].notna().sum()

        accuracy = n_correct / n_evaluated if n_evaluated > 0 else 0

        cum_ret = (1 + df['daily_pnl']).prod() - 1
        daily_rets = df['daily_pnl'].values
        ann_ret = daily_rets.mean() * 252 if len(daily_rets) > 0 else 0
        ann_vol = daily_rets.std() * np.sqrt(252) if len(daily_rets) > 0 else 0
        sharpe = ann_ret / (ann_vol + 1e-9)

        # Max drawdown
        cum = (1 + df['daily_pnl']).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_dd = drawdown.min()

        # By regime
        by_regime = {}
        for regime_id in [0, 1, 2]:
            sub = df[df['market_regime'] == regime_id]
            sub_eval = sub[sub['is_correct'].notna()]
            if len(sub_eval) > 0:
                acc = sub_eval['is_correct'].sum() / len(sub_eval)
                by_regime[['正常', '高波动', '趋势'][regime_id]] = {
                    'accuracy': round(float(acc), 4),
                    'count': len(sub_eval),
                    'cum_return': round(float((1 + sub['daily_pnl']).prod() - 1), 4),
                }

        # By direction
        by_direction = {}
        for d in ['UP', 'DOWN']:
            sub = df[df['direction'] == d]
            sub_eval = sub[sub['is_correct'].notna()]
            if len(sub_eval) > 0:
                acc = sub_eval['is_correct'].sum() / len(sub_eval)
                dname = '做多' if d == 'UP' else '做空'
                by_direction[dname] = {
                    'accuracy': round(float(acc), 4),
                    'count': len(sub_eval),
                }

        self.backtest_result = {
            'total_days': len(df),
            'trades': int(n_trades),
            'evaluated': int(n_evaluated),
            'correct': int(n_correct),
            'accuracy': round(float(accuracy), 4),
            'cumulative_return': round(float(cum_ret), 4),
            'annualized_return': round(float(ann_ret), 4),
            'annualized_volatility': round(float(ann_vol), 4),
            'sharpe_ratio': round(float(sharpe), 4),
            'max_drawdown': round(float(max_dd), 4),
            'by_regime': by_regime,
            'by_direction': by_direction,
            'df_daily': df,
        }

        return self.backtest_result

    # ─── Comparison ─────────────────────────────────────────

    def compare_with_lightgbm(self, memory=None):
        """Compare LLM predictions with LightGBM daily signals from memory."""
        if not hasattr(self, 'backtest_result') or self.backtest_result is None:
            self.daily_backtest()

        # Load LightGBM daily signals from memory
        if memory is None:
            from memory import TradingMemory
            memory = TradingMemory(self.base_dir)

        lgb_records = memory._load_all()
        lgb_evaluated = [r for r in lgb_records if r.get('is_correct') is not None]

        lgb_total = len(lgb_evaluated)
        lgb_correct = sum(1 for r in lgb_evaluated if r['is_correct'])
        lgb_acc = lgb_correct / lgb_total if lgb_total > 0 else 0

        # LGB simple daily P&L (direction × daily_return from close-to-close)
        lgb_trades = [r for r in lgb_records if r.get('direction', 0) != 0]
        lgb_pnl = 0.0
        for r in lgb_trades:
            actual = r.get('actual_return')
            if actual is not None:
                lgb_pnl += r['direction'] * actual / 100
        lgb_cum_ret = lgb_pnl  # approximate

        llm = self.backtest_result

        # By regime comparison
        regime_compare = []
        for regime_name in ['正常', '高波动', '趋势']:
            llm_regime = llm.get('by_regime', {}).get(regime_name, {})
            lgb_regime_records = [r for r in lgb_evaluated
                                  if r.get('regime_name') == regime_name]
            lgb_regime_acc = (sum(1 for r in lgb_regime_records if r['is_correct']) /
                              len(lgb_regime_records)) if lgb_regime_records else 0

            regime_compare.append({
                'regime': regime_name,
                'LLM_accuracy': llm_regime.get('accuracy', 0),
                'LGB_accuracy': round(float(lgb_regime_acc), 4),
                'LLM_count': llm_regime.get('count', 0),
                'LGB_count': len(lgb_regime_records),
            })

        comparison = {
            'llm': {
                'accuracy': llm['accuracy'],
                'cumulative_return': llm['cumulative_return'],
                'sharpe': llm['sharpe_ratio'],
                'max_drawdown': llm['max_drawdown'],
                'trades': llm['trades'],
            },
            'lightgbm': {
                'accuracy': round(float(lgb_acc), 4),
                'cumulative_return': round(float(lgb_cum_ret), 4),
                'trades': len(lgb_trades),
                'evaluated': lgb_total,
            },
            'by_regime': regime_compare,
        }

        self.comparison = comparison
        return comparison

    def print_comparison(self):
        """Print a formatted comparison report."""
        if not hasattr(self, 'comparison') or self.comparison is None:
            self.compare_with_lightgbm()

        c = self.comparison

        print("\n" + "=" * 56)
        print("  DeepSeek LLM vs LightGBM 日频预测对比")
        print("=" * 56)

        print(f"\n{'指标':<20} {'DeepSeek LLM':>16} {'LightGBM':>16}")
        print("-" * 52)
        print(f"{'方向准确率':<20} {c['llm']['accuracy']:>15.2%} {c['lightgbm']['accuracy']:>15.2%}")
        print(f"{'累计收益':<20} {c['llm']['cumulative_return']:>15.2%} {c['lightgbm']['cumulative_return']:>15.2%}")
        print(f"{'夏普比率':<20} {c['llm']['sharpe']:>15.2f} {'—':>15}")
        print(f"{'最大回撤':<20} {c['llm']['max_drawdown']:>15.2%} {'—':>15}")
        print(f"{'交易次数':<20} {c['llm']['trades']:>15} {c['lightgbm']['trades']:>15}")

        print(f"\n--- 分市场状态准确率 ---")
        print(f"{'状态':<10} {'DeepSeek LLM':>16} {'LightGBM':>16}")
        print("-" * 42)
        for rc in c['by_regime']:
            print(f"{rc['regime']:<10} {rc['LLM_accuracy']:>15.2%} {rc['LGB_accuracy']:>15.2%}")

        print("=" * 56)

    def save_results(self):
        """Save LLM predictions and backtest to outputs/."""
        pred_path = os.path.join(self.base_dir, "outputs/llm_predictions.csv")
        bt_path = os.path.join(self.base_dir, "outputs/llm_backtest.json")

        if hasattr(self, 'df_llm_pred') and self.df_llm_pred is not None:
            self.df_llm_pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
            print(f"[LLM Predict] 预测已保存: {pred_path}")

        if hasattr(self, 'backtest_result') and self.backtest_result is not None:
            bt_out = {k: v for k, v in self.backtest_result.items()
                      if k != 'df_daily'}
            with open(bt_path, 'w', encoding='utf-8') as f:
                json.dump(bt_out, f, ensure_ascii=False, indent=2, default=str)
            print(f"[LLM Predict] 回测已保存: {bt_path}")


def run_experiment(base_dir, start_idx=0, end_idx=None, delay=0.5):
    """Run full LLM prediction experiment."""
    predictor = LLMDirectPredictor(base_dir)

    # Predict test period
    df = predictor.predict_test_period(start_idx=start_idx, end_idx=end_idx, delay=delay)

    # Backtest
    bt = predictor.daily_backtest()
    print(f"\n[LLM Predict] 回测完成:")
    print(f"  准确率: {bt['accuracy']:.2%} ({bt['correct']}/{bt['evaluated']})")
    print(f"  累计收益: {bt['cumulative_return']:.2%}")
    print(f"  夏普: {bt['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {bt['max_drawdown']:.2%}")

    # Compare with LightGBM
    predictor.compare_with_lightgbm()
    predictor.print_comparison()

    # Save
    predictor.save_results()

    return predictor


if __name__ == '__main__':
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else r"D:\桌面\F_Agent"
    run_experiment(base)
