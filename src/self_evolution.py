# -*- coding: utf-8 -*-
# self_evolution.py — 自我进化引擎 (LoRA-inspired Incremental Residual Boosting)
#
# 核心理念: 冻结基模型, 每周训练小型"适配器"学习残差, 每两月全量重训练
# 数学类比: LoRA的 W' = W + A×B
#           → 树模型的 f'(x) = f_base(x) + g_adapter(x)
#           → adapter = 极小型LightGBM (≈ 低秩矩阵的低参数量)
#
# 架构:
#   FrozenBaseModel (双月全量重训练)
#       +
#   AdapterStack (每周增量适配器, 时间衰减叠加)
#       +
#   FeedbackAnalyzer (交易反馈 → 训练权重 → 驱动适配方向)
#
# 周期:
#   Weekly: 训练新adapter(20棵树, max_depth=2) → 推入stack → 评估IC提升
#   Bimonthly: 吸收adapter经验 → 全量重训练base → 清空stack

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class AdapterRecord:
    """单个适配器的元数据"""
    adapter_id: str                          # e.g. "adapter_2026W23"
    trained_at: str                          # ISO timestamp
    n_trees: int
    max_depth: int
    n_features: int
    train_ic: float                          # adapter在训练期的IC
    eval_ic: float                           # adapter在验证期的IC
    residual_std: float                      # 残差标准差
    decay_weight: float = 1.0                # 时间衰减权重
    feedback_summary: Dict = field(default_factory=dict)
    # 适配器专注的市场状态
    focus_regime: Optional[int] = None       # 0=Normal, 1=HighVol, 2=Trend
    focus_direction: Optional[int] = None    # 1=做多, -1=做空


@dataclass
class EvolutionReport:
    """自我进化报告"""
    timestamp: str
    cycle_type: str                          # 'weekly' or 'bimonthly'
    
    # Base model metrics
    base_ic: float
    base_ic_by_regime: Dict[int, float] = field(default_factory=dict)
    
    # Adapter metrics
    new_adapter: Optional[AdapterRecord] = None
    adapter_stack_size: int = 0
    adapter_contribution_ic: float = 0.0     # adapter带来的IC提升
    
    # Combined metrics
    combined_ic: float = 0.0
    ic_improvement: float = 0.0
    
    # Feedback analysis
    feedback: Dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Adapter decay weights (for ensemble prediction)
    active_adapters: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Feedback Analyzer — 从交易记忆中提取反馈信号驱动适配方向
# ═══════════════════════════════════════════════════════════════

class FeedbackAnalyzer:
    """分析交易记忆反馈, 生成适配器训练的样本权重和聚焦策略"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
    
    def load_memory(self) -> List[Dict]:
        memory_path = os.path.join(self.base_dir, "outputs/trade_memory.jsonl")
        if not os.path.exists(memory_path):
            return []
        records = []
        with open(memory_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
    
    def load_predictions(self) -> Optional[pd.DataFrame]:
        pred_path = os.path.join(self.base_dir, "outputs/df_predictions.pkl")
        if not os.path.exists(pred_path):
            return None
        return pd.read_pickle(pred_path)
    
    def analyze(self, window_days: int = 30) -> Dict:
        """
        综合分析最近N天的交易反馈。
        
        Returns:
            {
                'regime_accuracy': {0: 0.65, 1: 0.45, 2: 0.30},
                'direction_accuracy': {1: 0.60, -1: 0.55},
                'failure_hotspots': [(regime, direction, fail_rate), ...],
                'sample_weights': np.array (与训练数据对齐),
                'focus_regime': int or None (最需要适配的市场状态),
                'focus_direction': int or None (最需要适配的方向),
                'recent_accuracy_trend': [0.62, 0.58, 0.55, ...],
                'drift_severity': float (0-1, 漂移严重程度),
                'reflection_text': str (LLM或规则生成的反思文本),
            }
        """
        now = datetime.now()
        cutoff = now - timedelta(days=window_days)
        
        records = self.load_memory()
        evaluated = [
            r for r in records 
            if r.get('is_correct') is not None
        ]
        
        # Filter recent
        recent = [
            r for r in evaluated
            if r.get('trade_dt', '') >= cutoff.strftime('%Y-%m-%d')
        ]
        
        if len(recent) < 20:
            # Not enough recent data, use all evaluated
            recent = evaluated
        
        result = {
            'regime_accuracy': {},
            'direction_accuracy': {},
            'failure_hotspots': [],
            'focus_regime': None,
            'focus_direction': None,
            'recent_accuracy_trend': [],
            'drift_severity': 0.0,
            'reflection_text': '',
            'total_recent': len(recent),
            'total_evaluated': len(evaluated),
        }
        
        if not recent:
            return result
        
        # --- Regime accuracy ---
        for regime_id, rname in [(0, '正常'), (1, '高波动'), (2, '趋势')]:
            subset = [r for r in recent if r.get('market_regime') == regime_id]
            if len(subset) >= 5:
                acc = sum(1 for r in subset if r['is_correct']) / len(subset)
                result['regime_accuracy'][regime_id] = round(acc, 4)
        
        # --- Direction accuracy ---
        for d in [1, -1]:
            subset = [r for r in recent if r.get('direction') == d]
            if len(subset) >= 5:
                acc = sum(1 for r in subset if r['is_correct']) / len(subset)
                result['direction_accuracy'][d] = round(acc, 4)
        
        # --- Failure hotspots (regime × direction cross) ---
        for regime_id in [0, 1, 2]:
            for d in [1, -1]:
                subset = [r for r in recent 
                          if r.get('market_regime') == regime_id and r.get('direction') == d]
                if len(subset) >= 5:
                    acc = sum(1 for r in subset if r['is_correct']) / len(subset)
                    if acc < 0.5:  # Below chance → hotspot
                        result['failure_hotspots'].append({
                            'regime': regime_id,
                            'direction': d,
                            'accuracy': round(acc, 4),
                            'samples': len(subset),
                        })
        
        # --- Identify focus regime ---
        # The regime with the lowest accuracy that has enough samples gets priority
        regime_scores = []
        for rid, acc in result['regime_accuracy'].items():
            regime_scores.append((rid, acc))
        if regime_scores:
            worst_regime = min(regime_scores, key=lambda x: x[1])
            if worst_regime[1] < 0.55:
                result['focus_regime'] = worst_regime[0]
        
        # --- Identify focus direction ---
        dir_acc = result['direction_accuracy']
        if 1 in dir_acc and -1 in dir_acc:
            if abs(dir_acc[1] - dir_acc[-1]) > 0.15:
                result['focus_direction'] = 1 if dir_acc[1] < dir_acc[-1] else -1
        
        # --- Accuracy trend (7-day rolling windows) ---
        recent_sorted = sorted(recent, key=lambda r: r.get('trade_dt', ''))
        if len(recent_sorted) >= 14:
            trend = []
            for i in range(7, len(recent_sorted) + 1, 3):
                window = recent_sorted[max(0, i-7):i]
                acc = sum(1 for r in window if r['is_correct']) / len(window)
                trend.append(round(acc, 4))
            result['recent_accuracy_trend'] = trend
        
        # --- Drift severity ---
        # If accuracy trend is declining, compute severity
        if len(result['recent_accuracy_trend']) >= 3:
            start_acc = result['recent_accuracy_trend'][0]
            end_acc = result['recent_accuracy_trend'][-1]
            decline = start_acc - end_acc
            # Normalize to 0-1 range
            result['drift_severity'] = round(max(0, min(1, decline * 5)), 4)
        
        # --- Reflection text (rule-based) ---
        lines = []
        lines.append(f"=== 反馈分析 ({window_days}天窗口) ===")
        lines.append(f"近期记录: {len(recent)} 笔 (总评估: {len(evaluated)} 笔)")
        
        overall_acc = sum(1 for r in recent if r['is_correct']) / len(recent)
        lines.append(f"近期准确率: {overall_acc:.2%}")
        
        for rid, acc in result['regime_accuracy'].items():
            rname = ['正常', '高波动', '趋势'][rid]
            lines.append(f"  {rname}市: {acc:.2%}")
        
        if result['failure_hotspots']:
            lines.append(f"\n失败热点:")
            for hs in result['failure_hotspots']:
                rname = ['正常', '高波动', '趋势'][hs['regime']]
                dname = {1: '做多', -1: '做空'}[hs['direction']]
                lines.append(f"  {rname}×{dname}: {hs['accuracy']:.2%} (n={hs['samples']})")
        
        if result['focus_regime'] is not None:
            rname = ['正常', '高波动', '趋势'][result['focus_regime']]
            lines.append(f"\n适配器焦点: {rname}市")
        
        if result['drift_severity'] > 0.3:
            lines.append(f"⚠ 制度漂移严重度: {result['drift_severity']:.2f}")
        
        result['reflection_text'] = '\n'.join(lines)
        
        return result
    
    def generate_sample_weights(
        self, 
        df: pd.DataFrame, 
        feedback: Dict, 
        base_half_life: int = 30
    ) -> np.ndarray:
        """
        基于反馈生成训练样本权重。
        
        权重规则:
        1. 时间衰减 (基础权重): 越近的样本权重越高
        2. 失败热点加权: 属于失败热点(regime×direction)的样本权重×2
        3. 制度漂移加权: 如果检测到严重漂移, 近期样本权重×3
        
        Args:
            df: 训练数据 (需有 'date' 或 'trade_dt' 列)
            feedback: analyze() 的输出
            base_half_life: 基础时间衰减半衰期(天)
        
        Returns:
            sample_weight: np.ndarray, shape=(len(df),)
        """
        n = len(df)
        weights = np.ones(n)
        
        # 1. 时间衰减
        if 'date' in df.columns:
            newest = df['date'].max()
            age_days = (newest - df['date']).dt.total_seconds() / 86400
            weights *= np.exp(-np.log(2) * np.clip(age_days, 0, None) / base_half_life)
        
        # 2. 失败热点加权
        if feedback.get('failure_hotspots') and 'Market_Regime' in df.columns:
            for hs in feedback['failure_hotspots']:
                regime_id = hs['regime']
                # We don't have direction in training data, but we can weight by regime
                mask = (df['Market_Regime'] == regime_id)
                weights[mask] *= 2.0
        
        # 3. 制度漂移加权
        drift_severity = feedback.get('drift_severity', 0)
        if drift_severity > 0.3 and 'date' in df.columns:
            # Give extra weight to last 14 days
            newest = df['date'].max()
            recent_mask = (newest - df['date']).dt.days <= 14
            boost = 1 + drift_severity * 2  # 1.6x to 3x
            weights[recent_mask] *= boost
        
        # Normalize to mean=1
        if weights.sum() > 0:
            weights = weights / weights.mean()
        
        return weights


# ═══════════════════════════════════════════════════════════════
# FrozenBaseModel — 冻结的基模型包装器
# ═══════════════════════════════════════════════════════════════

class FrozenBaseModel:
    """包装现有LightGBM双模型, 提供冻结预测接口"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.model_path = os.path.join(base_dir, "models/trained_model.pkl")
        self._model_data = None
    
    def load(self) -> bool:
        """加载基模型。返回是否成功。"""
        if not os.path.exists(self.model_path):
            print(f"[FrozenBase] 模型文件不存在: {self.model_path}")
            return False
        
        with open(self.model_path, 'rb') as f:
            self._model_data = pickle.load(f)
        
        self.model_base = self._model_data['model_base']
        self.model_active = self._model_data.get('model_active')
        self.scaler = self._model_data['scaler']
        self.features = self._model_data['features']
        
        print(f"[FrozenBase] 已加载基模型: {len(self.features)} 特征")
        return True
    
    def predict(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """冻结预测 (不修改模型参数)"""
        X_scaled = self.scaler.transform(X)
        
        preds = self.model_base.predict(X_scaled)
        
        # Active model for HighVol/Trend regimes
        if self.model_active is not None and regimes is not None:
            active_mask = np.isin(regimes, [1, 2])
            if active_mask.sum() > 0:
                preds[active_mask] = self.model_active.predict(X_scaled[active_mask])
        
        return preds
    
    @property
    def is_loaded(self) -> bool:
        return self._model_data is not None
    
    def get_feature_importance(self) -> pd.DataFrame:
        """返回基模型的特征重要性"""
        if not self.is_loaded:
            return pd.DataFrame()
        imp = pd.DataFrame({
            'feature': self.features,
            'importance': self.model_base.feature_importances_
        }).sort_values('importance', ascending=False)
        return imp


# ═══════════════════════════════════════════════════════════════
# ResidualAdapter — LoRA-like 轻量级残差适配器
# ═══════════════════════════════════════════════════════════════

class ResidualAdapter:
    """
    LoRA-inspired 残差适配器。
    
    类比: LoRA的 W' = W + A×B (A∈R^{d×r}, B∈R^{r×d})
    这里:    f'(x) = f_base(x) + g_adapter(x)
    
    g_adapter 是一个极小型的 LightGBM:
    - 20棵树 (vs base的200棵)
    - max_depth=2 (vs base的3-4)
    - 强正则化
    - 参数量 ≈ base的 1/20 ~ 1/30
    
    这种低容量设计类似于LoRA的低秩约束: adapter只能学习"方向性修正",
    不能颠覆基模型的预测逻辑。
    """
    
    def __init__(self, adapter_id: str):
        self.adapter_id = adapter_id
        self.model: Optional[LGBMRegressor] = None
        self.scaler: Optional[RobustScaler] = None
        self.features: List[str] = []
        self.metadata: Optional[AdapterRecord] = None
    
    def train(
        self,
        X_train: np.ndarray,
        residuals: np.ndarray,       # y_true - base_pred
        X_val: np.ndarray,
        residuals_val: np.ndarray,
        features: List[str],
        sample_weight: Optional[np.ndarray] = None,
        focus_regime: Optional[int] = None,
        focus_direction: Optional[int] = None,
    ) -> AdapterRecord:
        """
        训练残差适配器。
        
        Args:
            X_train, residuals: 训练数据 (残差 = Target_Ret - base_prediction)
            X_val, residuals_val: 验证数据
            features: 特征名列表
            sample_weight: 反馈驱动的样本权重
            focus_regime/direction: 适配器焦点 (来自反馈分析)
        """
        n_samples = len(X_train)
        
        # 极小型LGBM — 参数量控制在base的1/20
        self.model = LGBMRegressor(
            n_estimators=20,              # 极少量树
            learning_rate=0.01,
            num_leaves=4,                 # 极浅
            max_depth=2,                  # 极浅
            lambda_l1=20.0,              # 强正则化 (防止过拟合残差噪声)
            lambda_l2=20.0,
            feature_fraction=0.3,
            bagging_fraction=0.4,
            bagging_freq=3,
            min_child_samples=500,       # 高最小样本 (确保泛化)
            min_split_gain=0.02,
            objective='regression_l1',    # MAE — 对残差异常值鲁棒
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        
        self.model.fit(
            X_train, residuals,
            sample_weight=sample_weight,
            eval_set=[(X_val, residuals_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ]
        )
        
        # Evaluate
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        train_ic = np.corrcoef(residuals, train_preds)[0, 1] if len(residuals) > 1 else 0
        val_ic = np.corrcoef(residuals_val, val_preds)[0, 1] if len(residuals_val) > 1 else 0
        
        self.features = features
        self.metadata = AdapterRecord(
            adapter_id=self.adapter_id,
            trained_at=datetime.now().isoformat(),
            n_trees=self.model.best_iteration_ if self.model.best_iteration_ else 20,
            max_depth=2,
            n_features=len(features),
            train_ic=round(train_ic, 4),
            eval_ic=round(val_ic, 4),
            residual_std=round(float(np.std(residuals)), 6),
            decay_weight=1.0,
            focus_regime=focus_regime,
            focus_direction=focus_direction,
        )
        
        print(f"[Adapter] {self.adapter_id} 训练完成: "
              f"train_IC={train_ic:.4f}, val_IC={val_ic:.4f}, "
              f"residual_std={self.metadata.residual_std:.6f}")
        
        return self.metadata
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """适配器预测残差修正"""
        if self.model is None:
            return np.zeros(len(X))
        return self.model.predict(X)


# ═══════════════════════════════════════════════════════════════
# AdapterStack — 适配器堆栈管理
# ═══════════════════════════════════════════════════════════════

class AdapterStack:
    """
    管理多个适配器, 提供时间衰减加权组合预测。
    
    组合公式:
      final_pred = base_pred + Σ(adapter_i.pred × decay_weight_i)
    
    衰减权重:
      每周衰减为原来的 0.85 (半衰期 ≈ 4.5周)
      确保旧适配器的影响逐渐消退
    """
    
    DECAY_RATE = 0.85  # 每周衰减率
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.stack_path = os.path.join(base_dir, "models/adapter_stack.pkl")
        self.adapters: List[ResidualAdapter] = []
        self.records: List[AdapterRecord] = []
        self._load()
    
    def _load(self):
        """从磁盘加载适配器堆栈，损坏时自动备份"""
        if not os.path.exists(self.stack_path):
            return
        
        try:
            with open(self.stack_path, 'rb') as f:
                data = pickle.load(f)
            self.adapters = data.get('adapters', [])
            self.records = data.get('records', [])
            print(f"[AdapterStack] 加载 {len(self.adapters)} 个适配器")
        except Exception as e:
            print(f"[AdapterStack] 加载失败: {e}")
            # Backup corrupted file
            try:
                backup_path = self.stack_path + ".corrupted." + datetime.now().strftime("%Y%m%d_%H%M%S")
                os.rename(self.stack_path, backup_path)
                print(f"[AdapterStack] 损坏文件已备份: {backup_path}")
            except:
                try:
                    os.remove(self.stack_path)
                except:
                    pass
            self.adapters = []
            self.records = []
    
    def _save(self):
        """持久化适配器堆栈"""
        os.makedirs(os.path.dirname(self.stack_path), exist_ok=True)
        with open(self.stack_path, 'wb') as f:
            pickle.dump({
                'adapters': self.adapters,
                'records': self.records,
                'saved_at': datetime.now().isoformat(),
            }, f)
    
    def push(self, adapter: ResidualAdapter) -> AdapterRecord:
        """
        推入新适配器, 衰减所有旧适配器的权重。
        """
        # Decay existing adapters
        for rec in self.records:
            rec.decay_weight *= self.DECAY_RATE
        
        self.adapters.append(adapter)
        self.records.append(adapter.metadata)
        
        self._save()
        
        n = len(self.adapters)
        print(f"[AdapterStack] 推入 {adapter.adapter_id}, 堆栈大小: {n}")
        
        # Cleanup: 移除衰减到极低权重的旧适配器 (保留最多8个)
        if len(self.adapters) > 8:
            # Sort by decay_weight descending, keep top 8
            paired = list(zip(self.adapters, self.records))
            paired.sort(key=lambda x: x[1].decay_weight, reverse=True)
            self.adapters = [p[0] for p in paired[:8]]
            self.records = [p[1] for p in paired[:8]]
            self._save()
        
        return adapter.metadata
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        所有适配器的加权组合预测。
        """
        if not self.adapters:
            return np.zeros(len(X))
        
        combined = np.zeros(len(X))
        for adapter, record in zip(self.adapters, self.records):
            if record.decay_weight > 0.01:  # Skip nearly-zero adapters
                combined += adapter.predict(X) * record.decay_weight
        
        return combined
    
    def get_active_adapters(self) -> List[Dict]:
        """获取活跃适配器摘要"""
        return [
            {
                'id': r.adapter_id,
                'trained_at': r.trained_at[:19],
                'decay_weight': round(r.decay_weight, 4),
                'eval_ic': r.eval_ic,
                'focus_regime': r.focus_regime,
                'focus_direction': r.focus_direction,
            }
            for r in self.records
            if r.decay_weight > 0.01
        ]
    
    def clear(self):
        """清空适配器堆栈 (双月全量重训练后)"""
        self.adapters = []
        self.records = []
        if os.path.exists(self.stack_path):
            os.remove(self.stack_path)
        print("[AdapterStack] 堆栈已清空")
    
    @property
    def size(self) -> int:
        return len(self.adapters)


# ═══════════════════════════════════════════════════════════════
# SelfEvolutionEngine — 自我进化主编排器
# ═══════════════════════════════════════════════════════════════

class SelfEvolutionEngine:
    """
    F_Agent 自我进化引擎。
    
    类比 LoRA 微调架构:
    
      LoRA (神经网络):
        W' = W + A×B
        ├── W: 冻结的预训练权重 (参数量大)
        ├── A∈R^{d×r}: 低秩矩阵 (参数量小)
        └── B∈R^{r×d}: 低秩矩阵 (参数量小)
    
      F_Agent (树模型):
        f'(x) = f_base(x) + Σ g_adapter_i(x) × w_i
        ├── f_base: 冻结的LightGBM (200棵树, 双月全量重训练)
        ├── g_adapter_i: 第i周适配器 (20棵树, 深度2)
        └── w_i: 时间衰减权重 (每周×0.85)
    
    周期:
      Weekly  (每周六): 分析反馈 → 训练适配器 → 推入堆栈
      Bimonthly (每两月): 吸收经验 → 全量重训练 → 清空堆栈
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.base_model = FrozenBaseModel(base_dir)
        self.adapter_stack = AdapterStack(base_dir)
        self.feedback_analyzer = FeedbackAnalyzer(base_dir)
        
        # Paths
        self.factors_path = os.path.join(base_dir, "outputs/df_factors.pkl")
        self.pred_path = os.path.join(base_dir, "outputs/df_predictions.pkl")
        self.evo_report_path = os.path.join(base_dir, "outputs/evolution_report.json")
        self.evo_history_path = os.path.join(base_dir, "outputs/evolution_history.jsonl")
    
    # ─── Weekly Adaptation ──────────────────────────────────
    
    def weekly_adapt(self) -> Optional[EvolutionReport]:
        """
        每周适配流程:
        1. 加载基模型 (frozen)
        2. 分析最近30天交易反馈
        3. 计算基模型残差
        4. 训练适配器 (学习残差)
        5. 评估适配器效果
        6. 推入堆栈
        7. 生成报告
        """
        print("\n" + "=" * 60)
        print("  F_Agent 自我进化 — 每周适配 (Weekly Adaptation)")
        print("=" * 60)
        
        # Step 1: Load base model
        if not self.base_model.load():
            print("[Evo] ❌ 无法加载基模型, 终止")
            return None
        
        # Step 2: Load factor data
        if not os.path.exists(self.factors_path):
            print(f"[Evo] ❌ 因子文件不存在: {self.factors_path}")
            return None
        
        df_factors = pd.read_pickle(self.factors_path)
        print(f"[Evo] 因子数据: {len(df_factors)} 行 × {len(df_factors.columns)} 列")
        
        # Step 3: Feedback analysis
        print("\n[Evo] --- 步骤 1/5: 反馈分析 ---")
        feedback = self.feedback_analyzer.analyze(window_days=30)
        print(feedback.get('reflection_text', '无反馈数据'))
        
        # Step 4: Prepare training data with residuals
        print("\n[Evo] --- 步骤 2/5: 准备训练数据 ---")
        df_model, features = self._prepare_features(df_factors)
        if df_model is None or len(df_model) < 100:
            print("[Evo] ❌ 训练数据不足")
            return None
        
        # Compute base model predictions
        X_all = df_model[features].values
        regimes_all = df_model['Market_Regime'].values if 'Market_Regime' in df_model.columns else None
        base_preds = self.base_model.predict(X_all, regimes_all)
        y_all = df_model['Target_Ret'].values
        
        # Residuals = actual - predicted
        residuals = y_all - base_preds
        
        print(f"[Evo] 基模型整体IC: {np.corrcoef(y_all, base_preds)[0,1]:.4f}")
        print(f"[Evo] 残差标准差: {np.std(residuals):.6f}")
        
        # Time-based split: use last 20% as validation
        n = len(df_model)
        split_idx = int(n * 0.8)
        
        X_train = X_all[:split_idx]
        residuals_train = residuals[:split_idx]
        regimes_train = regimes_all[:split_idx] if regimes_all is not None else None
        dates_train = df_model['date'].iloc[:split_idx]
        
        X_val = X_all[split_idx:]
        residuals_val = residuals[split_idx:]
        
        # Generate feedback-driven sample weights
        train_df_for_weights = df_model.iloc[:split_idx].copy()
        sample_weights = self.feedback_analyzer.generate_sample_weights(
            train_df_for_weights, feedback, base_half_life=30
        )
        
        # Step 5: Train adapter
        print("\n[Evo] --- 步骤 3/5: 训练适配器 ---")
        week_num = datetime.now().isocalendar()[1]
        adapter_id = f"adapter_{datetime.now().year}W{week_num:02d}"
        
        adapter = ResidualAdapter(adapter_id)
        adapter_record = adapter.train(
            X_train=X_train,
            residuals=residuals_train,
            X_val=X_val,
            residuals_val=residuals_val,
            features=features,
            sample_weight=sample_weights,
            focus_regime=feedback.get('focus_regime'),
            focus_direction=feedback.get('focus_direction'),
        )
        
        # Step 6: Evaluate combined prediction
        print("\n[Evo] --- 步骤 4/5: 评估组合预测 ---")
        
        adapter_preds_val = adapter.predict(X_val)
        
        # Existing stack prediction
        stack_preds_val = self.adapter_stack.predict(X_val)
        
        # Combined predictions
        combined_val = base_preds[split_idx:] + adapter_preds_val + stack_preds_val
        y_val = y_all[split_idx:]
        
        base_ic = np.corrcoef(y_val, base_preds[split_idx:])[0, 1]
        adapter_only_ic = np.corrcoef(residuals_val, adapter_preds_val)[0, 1] if len(residuals_val) > 1 else 0
        combined_ic = np.corrcoef(y_val, combined_val)[0, 1] if len(y_val) > 1 else 0
        
        print(f"  Base IC:        {base_ic:.4f}")
        print(f"  Adapter IC:     {adapter_only_ic:.4f} (on residuals)")
        print(f"  Combined IC:    {combined_ic:.4f}")
        print(f"  IC 变化:        {combined_ic - base_ic:+.4f}")
        
        # Step 7: Push adapter to stack
        print("\n[Evo] --- 步骤 5/5: 推入适配器堆栈 ---")
        self.adapter_stack.push(adapter)
        
        # Build report
        report = EvolutionReport(
            timestamp=datetime.now().isoformat(),
            cycle_type='weekly',
            base_ic=round(base_ic, 4),
            new_adapter=adapter_record,
            adapter_stack_size=self.adapter_stack.size,
            adapter_contribution_ic=round(adapter_only_ic, 4),
            combined_ic=round(combined_ic, 4),
            ic_improvement=round(combined_ic - base_ic, 4),
            feedback=feedback,
            active_adapters=self.adapter_stack.get_active_adapters(),
        )
        
        # By-regime IC analysis
        if regimes_all is not None:
            for regime_id in [0, 1, 2]:
                mask = (regimes_all[split_idx:] == regime_id)
                if mask.sum() > 10:
                    regime_ic = np.corrcoef(y_val[mask], combined_val[mask])[0, 1]
                    report.base_ic_by_regime[regime_id] = round(regime_ic, 4)
        
        # Recommendations
        if combined_ic < base_ic - 0.005:
            report.recommendations.append(
                "⚠ 适配器降低了IC, 建议检查反馈质量和训练数据"
            )
        if feedback.get('drift_severity', 0) > 0.5:
            report.recommendations.append(
                f"⚠ 制度漂移严重 ({feedback['drift_severity']:.2f}), "
                "建议提前触发全量重训练"
            )
        if combined_ic > base_ic + 0.003:
            report.recommendations.append(
                f"✓ 适配器有效提升IC {combined_ic - base_ic:+.4f}"
            )
        
        # Save reports
        self._save_report(report)
        
        print("\n" + "=" * 60)
        print(f"  每周适配完成: IC {base_ic:.4f} → {combined_ic:.4f} "
              f"({combined_ic - base_ic:+.4f})")
        print("=" * 60)
        
        return report
    
    # ─── Bimonthly Full Retrain ─────────────────────────────
    
    def bimonthly_retrain(self) -> Optional[EvolutionReport]:
        """
        双月全量重训练:
        1. 收集所有适配器的学习经验
        2. 分析适配器贡献 (哪些市场状态的修正最有效)
        3. 用适配器经验调整训练权重
        4. 全量重训练基模型
        5. 清空适配器堆栈
        """
        print("\n" + "=" * 60)
        print("  F_Agent 自我进化 — 双月全量重训练 (Bimonthly Retrain)")
        print("=" * 60)
        
        # Step 1: Analyze adapter learnings
        print("\n[Evo] --- 步骤 1/5: 收集适配器经验 ---")
        
        adapter_history = self.records_to_dataframe()
        if len(adapter_history) > 0:
            print(f"[Evo] 适配器历史: {len(adapter_history)} 个")
            for _, row in adapter_history.iterrows():
                print(f"  {row['adapter_id']}: IC={row['eval_ic']:.4f}, "
                      f"decay={row['decay_weight']:.3f}")
        
        # Step 2: Feedback analysis for retraining guidance
        print("\n[Evo] --- 步骤 2/5: 双月反馈分析 ---")
        feedback = self.feedback_analyzer.analyze(window_days=60)
        print(feedback.get('reflection_text', '无反馈数据'))
        
        # Step 3: Load data and compute adapter-corrected targets
        print("\n[Evo] --- 步骤 3/5: 准备重训练数据 ---")
        
        if not os.path.exists(self.factors_path):
            print(f"[Evo] ❌ 因子文件不存在")
            return None
        
        df_factors = pd.read_pickle(self.factors_path)
        df_model, features = self._prepare_features(df_factors)
        
        if df_model is None or len(df_model) < 500:
            print("[Evo] ❌ 重训练数据不足")
            return None
        
        # Get adapter predictions
        X_all = df_model[features].values
        regimes_all = df_model['Market_Regime'].values if 'Market_Regime' in df_model.columns else None
        
        adapter_preds = self.adapter_stack.predict(X_all)
        
        y_all = df_model['Target_Ret'].values
        
        # Step 4: Generate retraining sample weights
        # Incorporate adapter learnings: where adapters were confident, weight higher
        adapter_confidence = np.abs(adapter_preds) / (np.std(adapter_preds) + 1e-8)
        adapter_confidence = np.clip(adapter_confidence, 0.5, 3.0)
        
        base_weights = self.feedback_analyzer.generate_sample_weights(
            df_model, feedback, base_half_life=90
        )
        
        # Blend: base feedback weights × adapter confidence
        retrain_weights = base_weights * adapter_confidence
        retrain_weights = retrain_weights / retrain_weights.mean()
        
        # Step 5: Run full retrain
        print("\n[Evo] --- 步骤 4/5: 执行全量重训练 ---")
        
        try:
            from retrain_optimized import RetrainOptimizer
            retrainer = RetrainOptimizer(self.base_dir)
            
            # Override sample weights for the retrain
            # Note: retrain_optimized.py uses its own training logic
            # We call it and let it do its thing
            success = retrainer.run_full_retrain()
            
            if not success:
                print("[Evo] ⚠ retrain_optimized 返回失败, 尝试直接训练")
                success = self._direct_retrain(df_model, features, retrain_weights)
        except ImportError:
            print("[Evo] retrain_optimized 不可用, 使用直接重训练")
            success = self._direct_retrain(df_model, features, retrain_weights)
        
        if not success:
            print("[Evo] ❌ 重训练失败")
            return None
        
        # Reload base model
        self.base_model.load()
        
        # Step 6: Clear adapter stack
        print("\n[Evo] --- 步骤 5/5: 重置适配器堆栈 ---")
        old_stack_size = self.adapter_stack.size
        self.adapter_stack.clear()
        
        # Evaluate new base model
        base_preds_new = self.base_model.predict(X_all, regimes_all)
        new_base_ic = np.corrcoef(y_all, base_preds_new)[0, 1] if len(y_all) > 1 else 0
        
        # Build report
        report = EvolutionReport(
            timestamp=datetime.now().isoformat(),
            cycle_type='bimonthly',
            base_ic=round(new_base_ic, 4),
            adapter_stack_size=0,
            combined_ic=round(new_base_ic, 4),
            feedback=feedback,
            recommendations=[
                f"全量重训练完成, 基模型IC: {new_base_ic:.4f}",
                f"已吸收 {old_stack_size} 个适配器经验",
                "适配器堆栈已重置",
            ],
        )
        
        self._save_report(report)
        
        print("\n" + "=" * 60)
        print(f"  双月重训练完成: 基模型IC = {new_base_ic:.4f}, "
              f"已吸收 {old_stack_size} 个适配器")
        print("=" * 60)
        
        return report
    
    # ─── Helpers ────────────────────────────────────────────
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """准备训练特征，使用基模型存储的特征列表保证与scaler兼容"""
        try:
            df = df.copy()
            
            # 构造Target_Ret (30分钟前向收益)
            df['next_close'] = df['close'].shift(-1)
            df['future_close'] = df['close'].shift(-31)  # 30min horizon
            df['Target_Ret'] = (df['future_close'] / df['next_close'] - 1) * 100
            
            # 使用基模型的特征列表
            if self.base_model.is_loaded and self.base_model.features:
                model_features = self.base_model.features
                # 只保留数据中存在的特征
                features = [f for f in model_features if f in df.columns]
                missing = len(model_features) - len(features)
                if missing > 0:
                    print(f"[Evo] 基模型特征: {len(features)}/{len(model_features)} "
                          f"(缺失 {missing} 个)")
                else:
                    print(f"[Evo] 基模型特征: {len(features)} (全部匹配)")
            else:
                # Fallback: use dynamic feature detection
                from LightGBM_model import prepare_training_data
                df_model, features = prepare_training_data(df, prediction_horizon=30)
                return df_model, features
            
            # Drop rows with NaN target
            df_model = df.dropna(subset=['Target_Ret']).copy()
            
            # Fill missing feature values with 0
            for col in features:
                if col in df_model.columns:
                    df_model[col] = df_model[col].fillna(0)
            
            # Keep relevant columns (deduplicate)
            keep_cols_base = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret',
                             'Market_Regime', 'oi', 'volume', 'money']
            # Deduplicate: features may overlap with keep_cols_base
            seen = set(keep_cols_base)
            keep_cols = list(keep_cols_base)
            for f in features:
                if f not in seen:
                    keep_cols.append(f)
                    seen.add(f)
            keep_cols = [c for c in keep_cols if c in df_model.columns]
            df_model = df_model[keep_cols]
            
            print(f"[Evo] 训练数据: {len(df_model)} 行 × {len(features)} 特征")
            
            return df_model, features
        except Exception as e:
            print(f"[Evo] 特征准备失败: {e}")
            import traceback
            traceback.print_exc()
            return None, []
    
    def _direct_retrain(
        self, 
        df_model: pd.DataFrame, 
        features: List[str], 
        sample_weights: np.ndarray
    ) -> bool:
        """直接重训练 (当retrain_optimized不可用时的fallback)"""
        from LightGBM_model import train_model
        from sklearn.preprocessing import RobustScaler
        
        n = len(df_model)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_df = df_model.iloc[:train_end]
        val_df = df_model.iloc[train_end:val_end]
        test_df = df_model.iloc[val_end:]
        
        scaler = RobustScaler()
        X_train = scaler.fit_transform(train_df[features])
        X_val = scaler.transform(val_df[features])
        X_test = scaler.transform(test_df[features])
        
        y_train = train_df['Target_Ret'].values
        y_val = val_df['Target_Ret'].values
        y_test = test_df['Target_Ret'].values
        
        sw_train = sample_weights[:train_end]
        
        model_base = train_model(X_train, y_train, X_val, y_val, 'base', sw_train)
        
        preds = model_base.predict(X_test)
        ic = np.corrcoef(y_test, preds)[0, 1]
        print(f"[Evo] 重训练 IC: {ic:.4f}")
        
        # Save
        with open(os.path.join(self.base_dir, "models/trained_model.pkl"), 'wb') as f:
            pickle.dump({
                'model_base': model_base,
                'model_active': None,
                'scaler': scaler,
                'features': features,
            }, f)
        
        # Save predictions
        test_df_out = test_df.copy()
        test_df_out['Pred_Ret'] = preds
        self._save_predictions(test_df_out)
        
        return True
    
    def _save_predictions(self, df: pd.DataFrame):
        """保存预测结果"""
        output_cols = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 'Pred_Ret',
                       'oi', 'volume', 'money', 'Market_Regime']
        available_cols = [c for c in output_cols if c in df.columns]
        df[available_cols].to_pickle(self.pred_path)
        print(f"[Evo] 预测已保存: {self.pred_path}")
    
    def _save_report(self, report: EvolutionReport):
        """保存进化报告"""
        # Latest report (overwrite)
        report_dict = {
            'timestamp': report.timestamp,
            'cycle_type': report.cycle_type,
            'base_ic': report.base_ic,
            'base_ic_by_regime': report.base_ic_by_regime,
            'adapter_stack_size': report.adapter_stack_size,
            'adapter_contribution_ic': report.adapter_contribution_ic,
            'combined_ic': report.combined_ic,
            'ic_improvement': report.ic_improvement,
            'feedback': {
                'drift_severity': report.feedback.get('drift_severity', 0),
                'focus_regime': report.feedback.get('focus_regime'),
                'focus_direction': report.feedback.get('focus_direction'),
                'total_recent': report.feedback.get('total_recent', 0),
            },
            'active_adapters': report.active_adapters,
            'recommendations': report.recommendations,
        }
        
        if report.new_adapter:
            report_dict['new_adapter'] = {
                'id': report.new_adapter.adapter_id,
                'eval_ic': report.new_adapter.eval_ic,
                'train_ic': report.new_adapter.train_ic,
                'n_trees': report.new_adapter.n_trees,
                'decay_weight': report.new_adapter.decay_weight,
            }
        
        with open(self.evo_report_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        # History (append)
        with open(self.evo_history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(report_dict, ensure_ascii=False) + '\n')
        
        print(f"[Evo] 报告已保存: {self.evo_report_path}")
    
    def records_to_dataframe(self) -> pd.DataFrame:
        """适配器记录转DataFrame"""
        if not self.adapter_stack.records:
            return pd.DataFrame()
        
        rows = []
        for r in self.adapter_stack.records:
            rows.append({
                'adapter_id': r.adapter_id,
                'trained_at': r.trained_at,
                'eval_ic': r.eval_ic,
                'train_ic': r.train_ic,
                'decay_weight': r.decay_weight,
                'focus_regime': r.focus_regime,
                'focus_direction': r.focus_direction,
                'n_trees': r.n_trees,
            })
        return pd.DataFrame(rows)
    
    def get_status(self) -> Dict:
        """获取引擎当前状态 (供Dashboard使用)"""
        base_loaded = self.base_model.is_loaded
        
        # Load latest report
        report = {}
        if os.path.exists(self.evo_report_path):
            try:
                with open(self.evo_report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
            except:
                pass
        
        # Load history
        history = []
        if os.path.exists(self.evo_history_path):
            try:
                with open(self.evo_history_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            history.append(json.loads(line))
            except:
                pass
        
        return {
            'base_model_loaded': base_loaded,
            'adapter_stack_size': self.adapter_stack.size,
            'active_adapters': self.adapter_stack.get_active_adapters(),
            'latest_report': report,
            'history': history[-20:],  # Last 20 cycles
            'total_cycles': len(history),
        }
    
    def generate_report_text(self, report: EvolutionReport) -> str:
        """生成可读报告文本"""
        lines = [
            "=" * 55,
            f"  F_Agent 自我进化报告 — {report.cycle_type}",
            f"  时间: {report.timestamp[:19]}",
            "=" * 55,
            "",
            f"【基模型 IC】: {report.base_ic:.4f}",
        ]
        
        if report.base_ic_by_regime:
            for rid, rname in [(0, '正常'), (1, '高波动'), (2, '趋势')]:
                if rid in report.base_ic_by_regime:
                    lines.append(f"  {rname}市: {report.base_ic_by_regime[rid]:.4f}")
        
        lines.append("")
        
        if report.cycle_type == 'weekly' and report.new_adapter:
            a = report.new_adapter
            lines.append(f"【新适配器】 {a.adapter_id}")
            lines.append(f"  Train IC: {a.train_ic:.4f}")
            lines.append(f"  Eval IC:  {a.eval_ic:.4f}")
            lines.append(f"  决策树数: {a.n_trees}")
            if a.focus_regime is not None:
                rname = ['正常', '高波动', '趋势'][a.focus_regime]
                lines.append(f"  焦点市场: {rname}市")
        
        lines.append("")
        lines.append(f"【组合 IC】: {report.combined_ic:.4f}")
        lines.append(f"【IC 提升】: {report.ic_improvement:+.4f}")
        lines.append(f"【适配器堆栈】: {report.adapter_stack_size} 个活跃")
        
        if report.active_adapters:
            lines.append("")
            lines.append("活跃适配器:")
            for aa in report.active_adapters:
                lines.append(f"  {aa['id']}: weight={aa['decay_weight']:.3f}, "
                           f"IC={aa['eval_ic']:.4f}")
        
        if report.recommendations:
            lines.append("")
            lines.append("【建议】")
            for rec in report.recommendations:
                lines.append(f"  {rec}")
        
        lines.append("")
        lines.append("=" * 55)
        
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI Entry Points
# ═══════════════════════════════════════════════════════════════

def run_weekly_adapt(base_dir: str) -> int:
    """CLI: 运行每周适配"""
    engine = SelfEvolutionEngine(base_dir)
    report = engine.weekly_adapt()
    if report:
        print(engine.generate_report_text(report))
        return 0
    return 1


def run_bimonthly_retrain(base_dir: str) -> int:
    """CLI: 运行双月全量重训练"""
    engine = SelfEvolutionEngine(base_dir)
    report = engine.bimonthly_retrain()
    if report:
        print(engine.generate_report_text(report))
        return 0
    return 1


def run_status(base_dir: str) -> int:
    """CLI: 查看进化引擎状态"""
    engine = SelfEvolutionEngine(base_dir)
    status = engine.get_status()
    
    print("=" * 55)
    print("  F_Agent 自我进化引擎 — 状态")
    print("=" * 55)
    print(f"  基模型: {'✓ 已加载' if status['base_model_loaded'] else '✗ 未加载'}")
    print(f"  适配器堆栈: {status['adapter_stack_size']} 个")
    print(f"  总进化周期: {status['total_cycles']}")
    
    if status['active_adapters']:
        print("\n  活跃适配器:")
        for aa in status['active_adapters']:
            print(f"    {aa['id']}: weight={aa['decay_weight']:.3f}, IC={aa['eval_ic']:.4f}")
    
    if status['latest_report']:
        r = status['latest_report']
        print(f"\n  最新报告 ({r.get('cycle_type', '?')}):")
        print(f"    Base IC: {r.get('base_ic', 'N/A')}")
        print(f"    Combined IC: {r.get('combined_ic', 'N/A')}")
        print(f"    IC 提升: {r.get('ic_improvement', 'N/A')}")
    
    return 0


def run_combine_predict(base_dir: str) -> int:
    """
    CLI: 运行组合预测 (base + adapters) 并保存结果。
    这允许在inference阶段使用适配器增强预测。
    """
    engine = SelfEvolutionEngine(base_dir)
    
    if not engine.base_model.load():
        print("[Evo] ❌ 基模型不可用")
        return 1
    
    if not os.path.exists(engine.factors_path):
        print(f"[Evo] ❌ 因子文件不存在")
        return 1
    
    df = pd.read_pickle(engine.factors_path)
    df_model, features = engine._prepare_features(df)
    
    if df_model is None:
        return 1
    
    X = df_model[features].values
    regimes = df_model['Market_Regime'].values if 'Market_Regime' in df_model.columns else None
    
    # Base predictions
    base_preds = engine.base_model.predict(X, regimes)
    
    # Adapter corrections
    adapter_preds = engine.adapter_stack.predict(X)
    
    # Combined
    combined_preds = base_preds + adapter_preds
    
    print(f"[Evo] 基模型预测范围: [{base_preds.min():.6f}, {base_preds.max():.6f}]")
    print(f"[Evo] 适配器修正范围: [{adapter_preds.min():.6f}, {adapter_preds.max():.6f}]")
    print(f"[Evo] 组合预测范围:   [{combined_preds.min():.6f}, {combined_preds.max():.6f}]")
    
    # Save combined predictions
    df_out = df_model.copy()
    df_out['Pred_Ret'] = combined_preds
    df_out['Base_Pred'] = base_preds
    df_out['Adapter_Correction'] = adapter_preds
    
    output_cols = ['date', 'trade_dt', 'ticker', 'close', 'Target_Ret', 
                   'Pred_Ret', 'Base_Pred', 'Adapter_Correction',
                   'Market_Regime']
    available_cols = [c for c in output_cols if c in df_out.columns]
    df_out[available_cols].to_pickle(engine.pred_path)
    
    print(f"[Evo] 组合预测已保存: {engine.pred_path}")
    return 0


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python self_evolution.py <command> [base_dir]")
        print("  weekly     — 每周适配 (训练新适配器)")
        print("  bimonthly  — 双月全量重训练")
        print("  status     — 查看引擎状态")
        print("  predict    — 运行组合预测并保存")
        sys.exit(1)
    
    command = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else r"D:\桌面\F_Agent"
    
    commands = {
        'weekly': run_weekly_adapt,
        'bimonthly': run_bimonthly_retrain,
        'status': run_status,
        'predict': run_combine_predict,
    }
    
    if command not in commands:
        print(f"未知命令: {command}")
        sys.exit(1)
    
    sys.exit(commands[command](base))
