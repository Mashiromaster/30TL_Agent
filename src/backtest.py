# -*- coding: utf-8 -*-
# backtest.py 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Backtester:
    def __init__(self, df):
        self.df = df.copy()
        self.commission = 0.000023
        self.tick_size = 0.01
        self.slippage_ticks = 1.0
        self.target_vol = 0.12
        
        # ============================================================
        # 根据状态设置参数
        # ============================================================
        
        # 全局参数
        self.signal_smooth_span = 160  # 信号平滑跨度
        
        # 分状态参数
        self.params = {
            # 高波动市
            1: {
                'threshold_upper': 0.8,  # 阈值
                'threshold_lower': 0.2,
                'confirm_window': 25,     # 确认窗口
                'min_hold_period': 60,    # 最小持仓
                'cooldown_period': 140,    # 冷却期
                'position_weight': 0.8,   
            },
            # 趋势市
            2: {
                'threshold_upper': 0.80,
                'threshold_lower': 0.20,
                'confirm_window': 50,
                'min_hold_period':30,
                'cooldown_period': 60,
                'position_weight': 0.8,
            },
            # 正常市
            0: {
                'threshold_upper': 0.9, 
                'threshold_lower': 0.05,
                'confirm_window': 60,     
                'min_hold_period': 90,    
                'cooldown_period': 180,   
                'position_weight': 1.0,   
            },
        }

    def run_strategy(self):
        print("[Backtest] 执行分状态策略...")

        n = len(self.df)

        # === 1. 信号平滑 ===
        self.df['Pred_Smooth'] = self.df['Pred_Ret'].ewm(
            span=self.signal_smooth_span, adjust=False
        ).mean()

        # === 2. 分状态计算阈值 ===
        pred_lagged = self.df['Pred_Smooth'].shift(1)

        self.df['Upper_Q'] = np.nan
        self.df['Lower_Q'] = np.nan
        self.df['Trade_Weight'] = 0.3

        if 'Market_Regime' in self.df.columns:
            for regime_id, params in self.params.items():
                mask = (self.df['Market_Regime'] == regime_id)

                upper_q = pred_lagged.rolling(480, min_periods=100).quantile(params['threshold_upper'])
                lower_q = pred_lagged.rolling(480, min_periods=100).quantile(params['threshold_lower'])

                self.df.loc[mask, 'Upper_Q'] = upper_q[mask]
                self.df.loc[mask, 'Lower_Q'] = lower_q[mask]
                self.df.loc[mask, 'Trade_Weight'] = params['position_weight']
        else:
            self.df['Upper_Q'] = pred_lagged.rolling(480, min_periods=100).quantile(0.70)
            self.df['Lower_Q'] = pred_lagged.rolling(480, min_periods=100).quantile(0.30)
            self.df['Trade_Weight'] = 1.0

        self.df['Upper_Q'] = self.df['Upper_Q'].ffill()
        self.df['Lower_Q'] = self.df['Lower_Q'].ffill()

        # === 3. 原始信号 ===
        self.df['Raw_Signal'] = 0
        upper_break = self.df['Pred_Smooth'] > self.df['Upper_Q']
        lower_break = self.df['Pred_Smooth'] < self.df['Lower_Q']

        self.df.loc[upper_break, 'Raw_Signal'] = 1
        self.df.loc[lower_break, 'Raw_Signal'] = -1
        
        # === 4. 分状态信号确认 ===
        self.df['Signal_Confirmed'] = 0
        raw_signal_lagged = self.df['Raw_Signal'].shift(1)
        
        if 'Market_Regime' in self.df.columns:
            for regime_id, params in self.params.items():
                mask = (self.df['Market_Regime'] == regime_id)
                confirm_window = params['confirm_window']
                
                def confirm_signal(x, window):
                    if len(x) < window:
                        return 0
                    non_zero = x[x != 0]
                    if len(non_zero) < window * 0.7:
                        return 0
                    if non_zero.nunique() == 1:
                        return non_zero.iloc[-1]
                    return 0
                
                confirmed = raw_signal_lagged.rolling(
                    window=confirm_window, min_periods=confirm_window
                ).apply(lambda x: confirm_signal(x, confirm_window), raw=False)
                
                self.df.loc[mask, 'Signal_Confirmed'] = confirmed[mask]
        else:
            def confirm_signal_default(x):
                if len(x) < 20:
                    return 0
                non_zero = x[x != 0]
                if len(non_zero) < 14:
                    return 0
                if non_zero.nunique() == 1:
                    return non_zero.iloc[-1]
                return 0
            
            self.df['Signal_Confirmed'] = raw_signal_lagged.rolling(
                window=20, min_periods=20
            ).apply(confirm_signal_default, raw=False)
        
        self.df['Signal_Confirmed'] = self.df['Signal_Confirmed'].fillna(0)
        
        # === 5. 分状态持仓管理 ===
        position = np.zeros(n)
        position_weight = np.zeros(n)
        
        last_trade_idx = -1000
        current_position = 0
        current_weight = 0
        hold_start_idx = 0
        current_regime = 0
        
        for i in range(n):
            signal = self.df['Signal_Confirmed'].iloc[i]
            regime = int(self.df['Market_Regime'].iloc[i]) if 'Market_Regime' in self.df.columns else 0
            weight = self.df['Trade_Weight'].iloc[i]
            
            params = self.params.get(regime, self.params[0])
            min_hold = params['min_hold_period']
            cooldown = params['cooldown_period']
            
            in_cooldown = (i - last_trade_idx) < cooldown
            hold_time = i - hold_start_idx if current_position != 0 else min_hold
            can_change = hold_time >= min_hold
            
            # 市场状态切换时，允许调整仓位
            regime_changed = (regime != current_regime)
            
            if (not in_cooldown and can_change) or regime_changed:
                if signal != 0:
                    if signal != current_position or abs(weight - current_weight) > 0.1:
                        current_position = signal
                        current_weight = weight
                        last_trade_idx = i
                        hold_start_idx = i
                        current_regime = regime
                elif current_position != 0 and hold_time >= min_hold * 1.5:
                    # 信号消失，考虑平仓
                    current_position = 0
                    current_weight = 0
                    last_trade_idx = i
            
            position[i] = current_position * current_weight
            position_weight[i] = current_weight
        
        self.df['Position'] = position
        self.df['Position_Weight'] = position_weight
        
        # === 6. 收益计算 ===
        self.df['Asset_Ret'] = self.df['close'].pct_change().fillna(0)
        self.df['Trades'] = (self.df['Position'] - self.df['Position'].shift(1)).abs()
        
        rel_slippage = (self.slippage_ticks * self.tick_size) / self.df['close']
        self.df['Cost'] = self.df['Trades'] * (self.commission + rel_slippage)
        
        self.df['Net_Ret'] = self.df['Position'] * self.df['Asset_Ret'] - self.df['Cost']
        
        # === 7. 杠杆控制 ===
        roll_vol = (
            self.df['Net_Ret']
            .shift(2)
            .rolling(window=240*5, min_periods=240)
            .std() * np.sqrt(240*252)
        )
        
        # 根据市场状态调整目标波动率
        self.df['Target_Vol'] = self.target_vol
        if 'Market_Regime' in self.df.columns:
            self.df.loc[self.df['Market_Regime'] == 1, 'Target_Vol'] = self.target_vol * 1.5  # 高波动市加杠杆
            self.df.loc[self.df['Market_Regime'] == 0, 'Target_Vol'] = self.target_vol * 1  # 正常市降杠杆
            self.df.loc[self.df['Market_Regime'] == 2, 'Target_Vol'] = self.target_vol * 0.8  # 趋势市
        
        self.df['Leverage'] = (self.df['Target_Vol'] / roll_vol).fillna(1.0).clip(0.3, 2.0)
        self.df.loc[roll_vol > 0.25, 'Leverage'] = 0.3
        self.df.loc[roll_vol > 0.35, 'Leverage'] = 0.0
        
        self.df['Strategy_Ret'] = self.df['Net_Ret'] * self.df['Leverage']
        self.df['Cum_Ret'] = (1 + self.df['Strategy_Ret']).cumprod()
        
        # === 诊断信息 ===
        self._print_diagnostics()

    def _print_diagnostics(self):
        """打印诊断信息"""
        position_changes = (self.df['Position'].diff().abs() > 0).sum()
        trading_days = len(self.df['date'].dt.date.unique())
        
        long_pct = (self.df['Position'] > 0).mean()
        short_pct = (self.df['Position'] < 0).mean()
        flat_pct = (self.df['Position'] == 0).mean()
        
        full_pos = (self.df['Position'].abs() >= 0.9).mean()
        partial_pos = ((self.df['Position'].abs() > 0) & (self.df['Position'].abs() < 0.9)).mean()
        
        avg_leverage = self.df.loc[self.df['Position'] != 0, 'Leverage'].mean() if (self.df['Position'] != 0).any() else 0
        
        print(f"\n[Backtest] 换仓次数: {position_changes}")
        print(f"[Backtest] 交易日数: {trading_days}")
        print(f"[Backtest] 日均换仓: {position_changes/trading_days:.2f} 次")
        print(f"[Backtest] 做多: {long_pct:.1%}, 做空: {short_pct:.1%}, 空仓: {flat_pct:.1%}")
        print(f"[Backtest] 全仓: {full_pos:.1%}, 轻仓: {partial_pos:.1%}")
        print(f"[Backtest] 平均杠杆: {avg_leverage:.2f}x")
        
        if 'Market_Regime' in self.df.columns:
            print(f"\n[Backtest] 分市场状态表现:")
            for regime_id in [0, 1, 2]:
                mask = (self.df['Market_Regime'] == regime_id)
                if mask.sum() > 0:
                    regime_ret = self.df.loc[mask, 'Strategy_Ret'].sum()
                    regime_trades = (self.df.loc[mask, 'Trades'] > 0).sum()
                    regime_hold = (self.df.loc[mask, 'Position'] != 0).mean()
                    regime_name = ['正常', '高波动', '趋势'][regime_id]
                    print(f"  - {regime_name}市: 收益={regime_ret:.2%}, 换仓={regime_trades}次, 持仓率={regime_hold:.1%}")

    def compute_metrics(self):
        """计算回测指标"""
        df_daily = self.df.groupby(self.df['date'].dt.date).agg({
            'Strategy_Ret': 'sum',
            'Position': 'last',
            'Trades': 'sum'
        }).reset_index()
        df_daily.columns = ['date', 'ret', 'position', 'trades']
        
        df_daily['cum'] = (1 + df_daily['ret']).cumprod()
        
        total_return = df_daily['cum'].iloc[-1] - 1
        n_days = len(df_daily)
        ann_ret = (1 + total_return) ** (252 / n_days) - 1
        ann_vol = df_daily['ret'].std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / ann_vol if ann_vol != 0 else 0
        
        df_daily['peak'] = df_daily['cum'].cummax()
        df_daily['drawdown'] = (df_daily['cum'] - df_daily['peak']) / df_daily['peak']
        mdd = df_daily['drawdown'].min()
        
        # 回撤持续时间
        in_dd = df_daily['drawdown'] < -0.001
        dd_periods = []
        start = None
        for i in range(len(df_daily)):
            if in_dd.iloc[i] and start is None:
                start = i
            elif not in_dd.iloc[i] and start is not None:
                dd_periods.append(i - start)
                start = None
        if start is not None:
            dd_periods.append(len(df_daily) - start)
        max_dd_duration = max(dd_periods) if dd_periods else 0
        
        total_trades = df_daily['trades'].sum() / 2
        ann_turnover = total_trades * (252 / n_days)
        
        trading_days = (df_daily['ret'].abs() > 0.0001).sum()
        winning_days = (df_daily['ret'] > 0.0001).sum()
        win_rate = winning_days / trading_days if trading_days > 0 else 0
        
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
        
        active_days = (df_daily['position'].abs() > 0).sum()
        active_ratio = active_days / n_days
        
        print("\n" + "="*50)
        print("详细风险指标")
        print("="*50)
        print(f"测试期: {n_days} 天")
        print(f"持仓天数: {active_days} 天 ({active_ratio:.1%})")
        print(f"总收益: {total_return:.2%}")
        print(f"最大回撤: {mdd:.2%}")
        print(f"回撤持续: {max_dd_duration} 天")

        return {
            '累计收益': f"{total_return:.2%}",
            '年化收益': f"{ann_ret:.2%}",
            '年化波动': f"{ann_vol:.2%}",
            '夏普比率': f"{sharpe:.3f}",
            '最大回撤': f"{mdd:.2%}",
            '回撤持续': f"{max_dd_duration}天",
            '年化换手': f"{ann_turnover:.1f}x",
            '日胜率': f"{win_rate:.2%}",
            'Calmar比率': f"{calmar:.2f}",
            '持仓比例': f"{active_ratio:.1%}",
        }, df_daily

    def plot_report(self, df_daily, save_path):
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        axes[0].plot(df_daily['date'], df_daily['cum'], color='darkgreen', linewidth=2)
        axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[0].fill_between(df_daily['date'], 1, df_daily['cum'], 
                            where=df_daily['cum'] >= 1, color='green', alpha=0.2)
        axes[0].fill_between(df_daily['date'], 1, df_daily['cum'], 
                            where=df_daily['cum'] < 1, color='red', alpha=0.2)
        axes[0].set_title('Cumulative Return (V3.2 - IC-Based)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('NAV')
        axes[0].grid(alpha=0.3)
        
        axes[1].fill_between(df_daily['date'], df_daily['drawdown'], 0, color='red', alpha=0.5)
        axes[1].set_title(f'Drawdown (Max: {df_daily["drawdown"].min():.2%})')
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(alpha=0.3)
        
        colors = ['green' if r > 0 else 'red' for r in df_daily['ret']]
        axes[2].bar(df_daily['date'], df_daily['ret'], color=colors, alpha=0.7, width=1)
        axes[2].axhline(y=0, color='black', linewidth=0.5)
        axes[2].set_title('Daily Returns')
        axes[2].set_ylabel('Return')
        axes[2].grid(alpha=0.3)
        
        axes[3].fill_between(df_daily['date'], 0, df_daily['position'], 
                            where=df_daily['position'] > 0, color='green', alpha=0.5, label='Long')
        axes[3].fill_between(df_daily['date'], 0, df_daily['position'], 
                            where=df_daily['position'] < 0, color='red', alpha=0.5, label='Short')
        axes[3].axhline(y=0, color='black', linewidth=0.5)
        axes[3].set_ylim(-1.5, 1.5)
        axes[3].set_title('Position')
        axes[3].set_ylabel('Position')
        axes[3].legend()
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def run_process(base_dir):
    print("\n" + "="*50)
    print("STEP 4: 策略回测")
    print("="*50)
    
    PRED_FILE = os.path.join(base_dir, "outputs/df_predictions.pkl")
    METRICS_FILE = os.path.join(base_dir, "outputs/backtest_metrics.csv")
    
    if not os.path.exists(PRED_FILE):
        print(f"[Backtest ERROR] 找不到预测文件: {PRED_FILE}")
        return False
    
    df_pred = pd.read_pickle(PRED_FILE)
    print(f"[Backtest] 加载测试集数据: {len(df_pred)} 行")
    
    tester = Backtester(df_pred)
    tester.run_strategy()
    metrics, df_daily = tester.compute_metrics()
    
    print("\n" + "="*50)
    print("样本外回测表现")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:12s}: {v}")
    print("="*50)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_FILE, index=False, encoding='utf-8-sig')
    
    report_path = os.path.join(base_dir, "outputs/strategy_report.png")
    tester.plot_report(df_daily, report_path)
    
    return True