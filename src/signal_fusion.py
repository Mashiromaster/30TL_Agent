# -*- coding: utf-8 -*-
# signal_fusion.py — Agent 融合决策层 (V1.0)
#
# 核心创新：LightGBM 模型预测 + RAG 研报检索 + 交易记忆反思 + LLM 情境推理
#           → 综合调整后的信号 + 可审计的决策理由
#
# 架构:
#   ┌─────────────────────────────────────────────────────────┐
#   │                    SignalFusionEngine                    │
#   │                                                         │
#   │  输入:                                                   │
#   │  ① base_signal: LightGBM 原始信号 (方向/置信度/排名)      │
#   │  ② memory_stats: 交易记忆统计 (相似状态准确率/失败模式)    │
#   │  ③ rag_context: RAG 检索结果 (政策变动/研报信号)          │
#   │  ④ factor_alerts: 异常因子预警                           │
#   │                                                         │
#   │  输出:                                                   │
#   │  ① fused_signal: 调整后的信号 (方向/调权/置信度修正)       │
#   │  ② adjustment_reason: 每一步调整的可审计解释              │
#   │  ③ fusion_report: 完整决策文本 (供 Dashboard 展示)        │
#   └─────────────────────────────────────────────────────────┘
#
# 决策逻辑:
#   Layer 1 — 规则层 (memory_stats + factor_alerts)
#         基于历史统计和异常因子的确定性规则 → 调权/取消信号
#   Layer 2 — 检索层 (rag_context)
#         非结构化研报 → 提取判断信号 → 确认/削弱模型方向
#   Layer 3 — LLM层 (可选, DeepSeek)
#         综合以上所有信息做情境推理 → 最终判断 + 完整理由

import os
import sys
import io
import json
import tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import BASE_DIR
from intelligence_events import redact_sensitive_text

# ═══ Windows Streamlit 全局修复 ═══
# 仅在 Streamlit 环境下重定向 stdout/stderr 到 StringIO
# (避免 pipe handle → WriteConsoleW → OSError(22))
# CLI 环境下保持正常输出
import importlib.util as _importlib_util
if sys.platform == 'win32' and _importlib_util.find_spec('streamlit') is not None:
    try:
        # Only redirect if we're actually inside a Streamlit run
        import streamlit as _st
        if hasattr(_st, 'runtime') and _st.runtime.exists():
            sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', write_through=True)
            sys.stderr = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', write_through=True)
    except Exception:
        pass

# ============================================================
# 数据结构
# ============================================================

@dataclass
class FusedSignal:
    """融合后的最终信号"""
    # 模型原始信号
    raw_direction: int = 0          # 1=做多, -1=做空, 0=观望
    raw_confidence: float = 0.0
    pred_rank_pct: Optional[float] = None
    predicted_return: float = 0.0
    predicted_return_smooth: float = 0.0
    regime: str = '正常'
    model_used: str = 'base'

    # 融合调整
    adjusted_direction: int = 0
    adjusted_weight: float = 0.0    # 下调后的仓位权重
    proposed_direction: int = 0
    proposed_weight: float = 0.0
    shadow_only: bool = True
    confidence_modifier: float = 0.0 # 置信度修正 (-0.3 ~ +0.1)
    fusion_level: str = 'none'      # none / rule / rag / llm

    # 决策理由
    reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)

    # RAG 上下文
    rag_policy_signals: List[str] = field(default_factory=list)
    rag_contradictions: List[str] = field(default_factory=list)

    # LLM 最终判断 (可选)
    llm_judgment: str = ''
    llm_reasoning: str = ''

    # Jev 只做影子门控，绝不在本层改变方向或仓位。
    jev_gate_action: str = ''
    jev_gate_confidence: float = 0.0
    jev_gate_probabilities: Dict[str, float] = field(default_factory=dict)
    jev_evidence_sufficient: float = 0.0
    jev_model: str = ''

    # Base decision provenance.  Research layers may add evidence but cannot
    # silently detach the result from the deterministic strategy decision.
    strategy_id: str = ''
    strategy_fingerprint: str = ''
    model_release_id: str = ''
    decision_id: str = ''

    timestamp: str = ''

    def to_dict(self) -> dict:
        return {
            'raw_direction': self.raw_direction,
            'raw_confidence': self.raw_confidence,
            'pred_rank_pct': self.pred_rank_pct,
            'predicted_return': self.predicted_return,
            'predicted_return_smooth': self.predicted_return_smooth,
            'regime': self.regime,
            'model_used': self.model_used,
            'adjusted_direction': self.adjusted_direction,
            'adjusted_weight': self.adjusted_weight,
            'proposed_direction': self.proposed_direction,
            'proposed_weight': self.proposed_weight,
            'shadow_only': self.shadow_only,
            'confidence_modifier': self.confidence_modifier,
            'fusion_level': self.fusion_level,
            'reasons': self.reasons,
            'risk_flags': self.risk_flags,
            'supporting_evidence': self.supporting_evidence,
            'rag_policy_signals': self.rag_policy_signals,
            'rag_contradictions': self.rag_contradictions,
            'llm_judgment': self.llm_judgment,
            'llm_reasoning': self.llm_reasoning,
            'jev_gate_action': self.jev_gate_action,
            'jev_gate_confidence': self.jev_gate_confidence,
            'jev_gate_probabilities': self.jev_gate_probabilities,
            'jev_evidence_sufficient': self.jev_evidence_sufficient,
            'jev_model': self.jev_model,
            'strategy_id': self.strategy_id,
            'strategy_fingerprint': self.strategy_fingerprint,
            'model_release_id': self.model_release_id,
            'decision_id': self.decision_id,
            'timestamp': self.timestamp,
        }


# ============================================================
# 融合引擎
# ============================================================

class SignalFusionEngine:
    """
    Agent 融合决策引擎。

    使用:
        engine = SignalFusionEngine(base_dir)
        fused = engine.fuse(
            base_signal=signal_dict,
            memory=memory_instance,
            rag=rag_analyzer_instance,
        )
    """

    def __init__(
        self,
        base_dir=BASE_DIR,
        *,
        jev_client=None,
        jev_gate_mode=None,
        allow_trade_adjustment: bool = False,
        api_key=None,
    ):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        if allow_trade_adjustment:
            raise ValueError(
                "research fusion cannot modify the canonical trading decision"
            )
        self.allow_trade_adjustment = False

        # LLM 配置
        self.api_key = (
            str(api_key) if api_key is not None else self._load_api_key()
        )
        self.llm_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.llm_enabled = bool(self.api_key) and os.environ.get(
            'F_AGENT_FUSION_LLM', '1'
        ).strip().lower() in ('1', 'true', 'yes')

        self.jev_gate_mode = (
            jev_gate_mode
            if jev_gate_mode is not None
            else os.environ.get('F_AGENT_JEV_GATE_MODE', 'off')
        ).strip().lower()
        if self.jev_gate_mode not in ('off', 'shadow'):
            raise ValueError("F_AGENT_JEV_GATE_MODE only supports 'off' or 'shadow'")
        self.jev_client = jev_client
        if self.jev_gate_mode == 'shadow' and self.jev_client is None:
            from jev_client import JevClient

            self.jev_client = JevClient(base_dir)

    def _load_api_key(self):
        """从 .env 或环境变量加载 API Key"""
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            env_path = os.path.join(self.base_dir, ".env")
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('DEEPSEEK_API_KEY='):
                                key = line.split('=', 1)[1].strip().strip('"').strip("'")
                                break
                except Exception:
                    pass
        return key

    # ═══════════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════════

    def fuse(
        self,
        base_signal: dict,
        memory=None,
        rag=None,
        factor_alerts: list = None,
        use_llm: bool = True,
        persist: bool = True,
    ) -> FusedSignal:
        """
        融合决策主入口。

        Parameters
        ----------
        base_signal: LightGBM 模型输出的原始信号 dict
        memory: TradingMemory 实例 (可选)
        rag: RAGAnalyzer 实例 (可选)
        factor_alerts: 异常因子列表 (可选)
        use_llm: 是否调用 LLM 做最终推理

        Returns
        -------
        FusedSignal: 融合后的完整决策
        """
        self._validate_base_signal(base_signal)
        if factor_alerts is None:
            factor_alerts = []

        fused = FusedSignal(
            raw_direction=base_signal.get('direction', 0),
            raw_confidence=base_signal.get('confidence', 0),
            pred_rank_pct=base_signal.get('pred_rank_pct'),
            predicted_return=base_signal.get('predicted_return', 0),
            predicted_return_smooth=base_signal.get('predicted_return_smooth', 0),
            regime=base_signal.get('regime_name', '正常'),
            model_used=base_signal.get('model_used', 'base'),
            adjusted_direction=base_signal.get('direction', 0),
            adjusted_weight=base_signal.get('suggested_weight', 0),
            proposed_direction=base_signal.get('direction', 0),
            proposed_weight=base_signal.get('suggested_weight', 0),
            shadow_only=True,
            strategy_id=str(base_signal.get('strategy_id', '')),
            strategy_fingerprint=str(base_signal.get('strategy_fingerprint', '')),
            model_release_id=str(base_signal.get('model_release_id', '')),
            decision_id=str(base_signal.get('decision_id', '')),
            timestamp=base_signal.get('timestamp', datetime.now().isoformat()),
        )

        # ── Layer 1: 规则层 (记忆统计 + 异常因子) ──
        self._apply_memory_rules(fused, memory)
        self._apply_factor_rules(fused, factor_alerts)

        # ── Layer 2: RAG 检索层 ──
        if rag is not None and fused.adjusted_direction != 0:
            self._apply_rag_rules(fused, rag)

        # ── Jev 影子门控：仅记录判断，不参与本轮信号调整 ──
        if (
            self.jev_gate_mode == 'shadow'
            and self.jev_client is not None
            and fused.adjusted_direction != 0
        ):
            self._apply_jev_shadow(fused, factor_alerts)

        # ── Layer 3: LLM 推理层 ──
        if use_llm and self.llm_enabled and fused.adjusted_direction != 0:
            self._apply_llm_judgment(fused, memory, rag, factor_alerts)

        # RAG/LLM/memory output is research evidence.  It is recorded as a
        # proposal, while the executable decision remains the deterministic
        # SignalPolicy result.  This research module has no opt-in path to
        # mutate the executable direction or weight.
        fused.proposed_direction = fused.adjusted_direction
        fused.proposed_weight = fused.adjusted_weight
        fused.adjusted_direction = fused.raw_direction
        fused.adjusted_weight = float(base_signal.get('suggested_weight', 0))
        if (
            fused.proposed_direction != fused.raw_direction
            or fused.proposed_weight != fused.adjusted_weight
        ):
            fused.reasons.append(
                '[安全边界] 研究层建议仅作影子记录，未修改基础交易决策'
            )

        # 保存
        if persist:
            self._save_fusion_result(fused)

        return fused

    @staticmethod
    def _validate_base_signal(base_signal):
        if not isinstance(base_signal, dict):
            raise ValueError("base_signal must be a decision mapping")
        if not base_signal.get('decision_ready', True):
            raise ValueError("base_signal decision is not ready")
        required = (
            'timestamp', 'direction', 'suggested_weight', 'strategy_id',
            'strategy_fingerprint', 'model_release_id', 'decision_id',
        )
        missing = [name for name in required if base_signal.get(name) in (None, '')]
        if missing:
            raise ValueError(f"base_signal missing provenance: {missing}")
        direction = base_signal['direction']
        if direction not in (-1, 0, 1):
            raise ValueError("base_signal direction must be -1, 0, or 1")
        weight = float(base_signal['suggested_weight'])
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("base_signal suggested_weight must be finite in [0, 1]")

    def _apply_jev_shadow(self, fused: FusedSignal, factor_alerts: list):
        """记录 Jev 风险判断；影子模式没有修改交易状态的权限。"""
        state = {
            'existing_signal': {
                'direction': {1: 'long', -1: 'short', 0: 'flat'}.get(
                    fused.adjusted_direction, 'flat'
                ),
                'regime': fused.regime,
                'model_used': fused.model_used,
            },
            'deterministic_rule_findings': list(fused.reasons),
            'rag_policy_signals': list(fused.rag_policy_signals),
            'rag_contradictions': list(fused.rag_contradictions),
            'factor_alert_names': [
                alert[0] if isinstance(alert, (tuple, list)) and alert else str(alert)
                for alert in factor_alerts[:8]
            ],
            'constraint': (
                'This is a shadow audit. The answer cannot reverse a signal and must not '
                'assume information absent from this state.'
            ),
        }
        try:
            result = self.jev_client.assess_signal_gate(state)
            if not result:
                return
            fused.jev_gate_action = str(result.get('action', ''))
            fused.jev_gate_confidence = float(result.get('confidence', 0.0))
            fused.jev_gate_probabilities = {
                str(key): float(value)
                for key, value in result.get('probabilities', {}).items()
            }
            fused.jev_evidence_sufficient = float(
                result.get('evidence_sufficient', 0.0)
            )
            fused.jev_model = str(result.get('model', ''))
        except Exception as e:
            print(f"[Fusion] Jev 影子门控异常: {redact_sensitive_text(e)}")

    # ═══════════════════════════════════════════════════════════
    # Layer 1: 规则层
    # ═══════════════════════════════════════════════════════════

    def _apply_memory_rules(self, fused: FusedSignal, memory):
        """基于交易记忆的统计规则调整信号"""
        if memory is None:
            return

        stats = self._get_memory_stats(memory)
        if stats is None:
            fused.reasons.append('[记忆] 无历史数据，跳过记忆校准')
            return

        # 核心指标
        recent_acc = stats.get('recent_10_accuracy', 0.5)
        regime_acc = stats.get('current_regime_accuracy', 0.5)
        direction_acc = stats.get('current_direction_accuracy', 0.5)
        win_streak = stats.get('current_win_streak', 0)
        lose_streak = stats.get('current_lose_streak', 0)

        # 规则 1: 近期连续亏损 → 降低仓位
        if lose_streak >= 3:
            old_w = fused.adjusted_weight
            fused.adjusted_weight *= 0.5
            fused.confidence_modifier -= 0.15
            fused.fusion_level = 'rule'
            fused.reasons.append(
                f'[记忆·风险] 近{lose_streak}笔连续亏损 → 仓位减半 ({old_w:.1%}→{fused.adjusted_weight:.1%})'
            )

        # 规则 2: 当前市场状态 + 当前方向的准确率偏低
        if regime_acc < 0.40:
            old_w = fused.adjusted_weight
            fused.adjusted_weight *= 0.6
            fused.confidence_modifier -= 0.10
            fused.fusion_level = 'rule'
            fused.reasons.append(
                f'[记忆·状态] {fused.regime}市准确率仅{regime_acc:.0%} → 仓位×0.6'
            )

        if direction_acc < 0.35 and fused.adjusted_direction != 0:
            old_w = fused.adjusted_weight
            fused.adjusted_weight *= 0.4
            fused.confidence_modifier -= 0.20
            fused.fusion_level = 'rule'
            dir_name = '做多' if fused.adjusted_direction == 1 else '做空'
            fused.reasons.append(
                f'[记忆·方向] {dir_name}信号近期准确率仅{direction_acc:.0%} → 仓位×0.4'
            )
            fused.risk_flags.append(f'历史{dir_name}信号准确率异常低 ({direction_acc:.0%})')

        # 规则 3: 近期准确率整体 > 70% → 可适度加仓
        if recent_acc > 0.70 and lose_streak == 0:
            old_w = fused.adjusted_weight
            fused.adjusted_weight = min(old_w * 1.2, 1.0)
            fused.confidence_modifier += 0.05
            fused.reasons.append(
                f'[记忆·确认] 近10笔准确率{recent_acc:.0%} → 适度加仓 ({old_w:.1%}→{fused.adjusted_weight:.1%})'
            )
            fused.supporting_evidence.append(f'模型近期表现优秀 (近10笔: {recent_acc:.0%})')

        # 规则 4: 完全反方向信号 → 反转
        if fused.adjusted_direction != 0 and direction_acc < 0.20 and lose_streak >= 2:
            old_dir = fused.adjusted_direction
            fused.adjusted_direction = -old_dir
            if fused.adjusted_direction == 0:
                fused.adjusted_direction = 0
            dir_name_old = '做多' if old_dir == 1 else '做空'
            dir_name_new = '做多' if fused.adjusted_direction == 1 else ('做空' if fused.adjusted_direction == -1 else '观望')
            fused.reasons.append(
                f'[记忆·反转] {dir_name_old}信号近期准确率{direction_acc:.0%}且连亏{lose_streak}笔 → 反转为{dir_name_new}'
            )

    def _apply_factor_rules(self, fused: FusedSignal, factor_alerts: list):
        """基于异常因子的规则调整"""
        if not factor_alerts:
            return

        extreme_count = sum(1 for _, _, pct in factor_alerts if pct > 90 or pct < 10)

        # 规则: 多个因子处于极端位置 → 信号可靠性下降
        if extreme_count >= 3:
            old_w = fused.adjusted_weight
            fused.adjusted_weight *= 0.7
            fused.confidence_modifier -= 0.10
            fused.fusion_level = 'rule'
            fused.reasons.append(
                f'[因子·预警] {extreme_count}个核心因子处于极端位置 → 仓位×0.7'
            )
            fused.risk_flags.append(
                f'{extreme_count}个因子处于极端分位 (历史<10%或>90%)，信号可靠性降低'
            )

    # ═══════════════════════════════════════════════════════════
    # Layer 2: RAG 检索层
    # ═══════════════════════════════════════════════════════════

    def _apply_rag_rules(self, fused: FusedSignal, rag):
        """基于 RAG 知识库检索的策略信号"""
        try:
            dir_name = '做多' if fused.adjusted_direction == 1 else '做空'
            query = (
                f"当前TL国债期货信号为{dir_name}，市场状态{fused.regime}。"
                f"请检索近期是否有央行政策变动、利率调整、或重大宏观事件"
                f"可能影响30年期国债期货的{fused.regime}市走势。"
            )
            result = rag.query(query, top_k=5, as_of=fused.timestamp)
            if not result or 'answer' not in result:
                return

            answer = result.get('answer', '')
            sources = result.get('sources', [])

            # 简单规则: 在 RAG 回答中搜索方向性关键词
            bullish_keywords = ['降准', '降息', '宽松', '利多', '下行', '回落', '流动性充裕']
            bearish_keywords = ['加息', '紧缩', '利空', '上行', '通胀', '收紧', '流动性紧张']

            bullish_hits = sum(1 for kw in bullish_keywords if kw in answer)
            bearish_hits = sum(1 for kw in bearish_keywords if kw in answer)

            if bullish_hits > bearish_hits and bullish_hits >= 2:
                fused.rag_policy_signals.append(
                    f'RAG文本命中{bullish_hits}个偏多关键词 (降准/宽松/利多债券)'
                )
                if fused.adjusted_direction == 1:
                    fused.supporting_evidence.append('政策面与模型做多方向一致')
                elif fused.adjusted_direction == -1:
                    fused.rag_contradictions.append(
                        f'政策面偏多，与模型做空信号矛盾 → 建议下调仓位或观望'
                    )
                    fused.adjusted_weight *= 0.5
                    fused.confidence_modifier -= 0.15
                    fused.fusion_level = 'rag'
                    fused.reasons.append(
                        f'[RAG·矛盾] 政策面偏多 (降准/宽松) vs 模型做空 → 仓位减半'
                    )

            elif bearish_hits > bullish_hits and bearish_hits >= 2:
                fused.rag_policy_signals.append(
                    f'RAG文本命中{bearish_hits}个偏空关键词 (紧缩/利空/收益率上行)'
                )
                if fused.adjusted_direction == -1:
                    fused.supporting_evidence.append('政策面与模型做空方向一致')
                elif fused.adjusted_direction == 1:
                    fused.rag_contradictions.append(
                        f'政策面偏空，与模型做多信号矛盾 → 建议下调仓位或观望'
                    )
                    fused.adjusted_weight *= 0.5
                    fused.confidence_modifier -= 0.15
                    fused.fusion_level = 'rag'
                    fused.reasons.append(
                        f'[RAG·矛盾] 政策面偏空 (收紧/利空) vs 模型做多 → 仓位减半'
                    )

            # 记录来源
            if sources:
                top_sources = sources[:3]
                fused.rag_policy_signals.append(
                    f'参考来源: {", ".join(s.get("title", s.get("source", "?"))[:40] for s in top_sources)}'
                )

        except Exception as e:
            safe_error = redact_sensitive_text(e)
            fused.reasons.append(f'[RAG·错误] 检索失败: {safe_error}')
            print(f"[Fusion] RAG 检索异常: {safe_error}")

    # ═══════════════════════════════════════════════════════════
    # Layer 3: LLM 推理层
    # ═══════════════════════════════════════════════════════════

    def _apply_llm_judgment(
        self, fused: FusedSignal, memory, rag, factor_alerts: list
    ):
        """调用 DeepSeek 做最终综合推理判断"""
        if not self.api_key:
            fused.reasons.append('[LLM] API Key 未配置，跳过 AI 推理')
            return

        try:
            prompt = self._build_fusion_prompt(fused, memory, factor_alerts)
            response = self._call_llm(prompt)

            if not response:
                fused.reasons.append('[LLM] 调用失败，保持规则层判断')
                return

            fused.llm_judgment = response
            fused.llm_reasoning = response  # 完整输出
            fused.fusion_level = 'llm'

            # 从 LLM 输出中尝试提取方向和权重建议
            self._parse_llm_response(fused, response)

            fused.reasons.append(
                '[LLM] AI 已完成综合推理 — 结合模型信号+市场状态+量价因子+政策面'
            )

        except Exception as e:
            fused.reasons.append(f'[LLM·错误] {redact_sensitive_text(e)}')

    def _build_fusion_prompt(self, fused: FusedSignal, memory, factor_alerts: list):
        """构造 LLM 融合决策 Prompt"""
        parts = []

        # 1. 模型信号
        dir_name = {1: '做多', -1: '做空', 0: '观望'}.get(fused.adjusted_direction, '观望')
        parts.append(f"## 1. LightGBM 模型信号")
        parts.append(f"- 原始方向: {dir_name}")
        parts.append(f"- 置信度: {fused.raw_confidence:.1%}")
        rank_text = (
            f"{fused.pred_rank_pct:.1%}"
            if fused.pred_rank_pct is not None else "N/A"
        )
        parts.append(f"- 排名分位: {rank_text}")
        parts.append(f"- 预测收益 (平滑): {fused.predicted_return_smooth:+.4f}%")
        parts.append(f"- 市场状态: {fused.regime}")
        parts.append(f"- 使用模型: {fused.model_used}")

        # 2. 已执行的规则层调整
        if fused.reasons:
            parts.append(f"\n## 2. 已执行的规则调整")
            for r in fused.reasons:
                parts.append(f"- {r}")
        parts.append(f"- 调整后仓位: {fused.adjusted_weight:.1%}")

        # 3. 异常因子
        if factor_alerts:
            parts.append(f"\n## 3. 异常因子预警")
            for name, direction, pct in factor_alerts[:8]:
                parts.append(f"- {name}: 历史{pct:.0f}%分位 ({direction})")

        # 4. 记忆统计
        if memory:
            stats = self._get_memory_stats(memory)
            if stats:
                parts.append(f"\n## 4. 交易记忆统计")
                parts.append(f"- 近10笔准确率: {stats.get('recent_10_accuracy', 0):.0%}")
                parts.append(f"- 当前状态({fused.regime})准确率: {stats.get('current_regime_accuracy', 0):.0%}")
                parts.append(f"- 当前连胜: {stats.get('current_win_streak', 0)}")
                parts.append(f"- 当前连亏: {stats.get('current_lose_streak', 0)}")

        # 5. RAG 政策信号
        if fused.rag_policy_signals:
            parts.append(f"\n## 5. RAG 政策面信号")
            for s in fused.rag_policy_signals:
                parts.append(f"- {s}")
        if fused.rag_contradictions:
            parts.append(f"\n## 6. RAG 政策面矛盾")
            for s in fused.rag_contradictions:
                parts.append(f"- {s}")

        parts.append(f"\n## 7. 任务")
        parts.append(
            "请作为量化交易分析师，综合以上所有信息，给出对当前信号的最终判断。\n"
            "判断要点：\n"
            "1. 模型做多/做空信号是否与宏观政策面一致？\n"
            "2. 当前的异常因子和历史记忆是否暗示信号可能失效？\n"
            "3. 最终建议：维持/削弱/反转当前信号？仓位应调整到多少？\n"
            "请用 100-200 字的中文给出简洁判断。"
        )

        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """调用 DeepSeek API"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1",
            )
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是30年期国债期货(TL)量化交易的分析师。"
                            "你的职责是综合模型预测、市场状态、因子异常、政策面信号、"
                            "和历史交易记忆，给出简洁的融合判断。"
                            "你严格基于给定的事实分析，不编造信息。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.2,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Fusion] LLM 调用失败: {redact_sensitive_text(e)}")
            return ""

    def _parse_llm_response(self, fused: FusedSignal, response: str):
        """从 LLM 输出中提取方向/仓位建议"""
        response_lower = response.lower()

        # 方向信号
        if any(kw in response for kw in ['反转', '做空', '反向']):
            if fused.adjusted_direction == 1:
                if any(kw in response for kw in ['强烈', '明确', '建议反']):
                    fused.adjusted_direction = -1
                    fused.reasons.append('[LLM·判断] 建议反转为做空')

        # 仓位信号
        if '减仓' in response or '降低仓位' in response or '半仓' in response:
            fused.adjusted_weight *= 0.5
            fused.reasons.append('[LLM·判断] 建议降低仓位')
        elif '轻仓' in response or '小仓' in response:
            fused.adjusted_weight *= 0.3
            fused.reasons.append('[LLM·判断] 建议轻仓试探')
        elif '观望' in response and fused.adjusted_direction != 0:
            fused.adjusted_direction = 0
            fused.adjusted_weight = 0
            fused.reasons.append('[LLM·判断] 建议观望，不交易')

    # ═══════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════

    def _get_memory_stats(self, memory) -> Optional[dict]:
        """从 TradingMemory 提取关键统计"""
        try:
            stats = memory.reflection_stats()
            if 'error' in stats:
                return None

            # 提取当前状态和方向的准确率
            regime_acc = 0.5
            direction_acc = 0.5

            by_regime = stats.get('by_regime', {})
            if by_regime:
                # 从 by_regime_direction 交叉表提取更精确的数据
                cross = stats.get('by_regime_direction', {})
                if cross:
                    regime_name_map = {0: '正常', 1: '高波动', 2: '趋势'}
                    total_correct = 0
                    total_count = 0
                    dir_correct = 0
                    dir_count = 0
                    for regime, dirs in cross.items():
                        for dname, info in dirs.items():
                            if regime in regime_name_map.values():
                                total_correct += info['count'] * info['accuracy']
                                total_count += info['count']
                    if total_count > 0:
                        regime_acc = total_correct / total_count

            by_dir = stats.get('by_direction', {})
            if by_dir:
                total_d = sum(v['count'] for v in by_dir.values())
                if total_d > 0:
                    direction_acc = sum(
                        v['count'] * v['accuracy'] for v in by_dir.values()
                    ) / total_d

            return {
                'total_records': stats.get('total_records', 0),
                'overall_accuracy': stats.get('overall_accuracy', 0.5),
                'recent_10_accuracy': stats.get('recent_10_accuracy', 0.5),
                'current_regime_accuracy': regime_acc,
                'current_direction_accuracy': direction_acc,
                'current_win_streak': stats.get('current_win_streak', 0),
                'current_lose_streak': 0,  # TradingMemory 目前没有连亏统计
                'max_win_streak': stats.get('max_win_streak', 0),
            }
        except Exception as e:
            print(f"[Fusion] Memory stats error: {redact_sensitive_text(e)}")
            return None

    def _save_fusion_result(self, fused: FusedSignal):
        """保存融合结果到 JSON"""
        try:
            path = os.path.join(self.output_dir, "fusion_signal.json")
            self._atomic_json(path, fused.to_dict())

            # 追加到历史
            hist_path = os.path.join(self.output_dir, "fusion_history.json")
            history = []
            if os.path.exists(hist_path):
                try:
                    with open(hist_path, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except Exception:
                    history = []
            history.append(fused.to_dict())
            history = history[-100:]  # 最多保留100条
            self._atomic_json(hist_path, history)

        except Exception as e:
            print(f"[Fusion] 保存失败: {redact_sensitive_text(e)}")

    @staticmethod
    def _atomic_json(path, payload):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
        )
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def get_history(self, n=20) -> list:
        """获取融合历史"""
        hist_path = os.path.join(self.output_dir, "fusion_history.json")
        if not os.path.exists(hist_path):
            return []
        try:
            with open(hist_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            return history[-n:]
        except Exception:
            return []


# ============================================================
# 便捷函数
# ============================================================

def run_fusion(
    base_dir=BASE_DIR,
    base_signal: dict = None,
    use_llm: bool = True,
    allow_trade_adjustment: bool = False,
    persist: bool = True,
    api_key=None,
) -> Optional[FusedSignal]:
    """
    一键融合决策。

    ``base_signal`` 必须由当前 ModelRuntime/SignalPolicy 调用显式传入；
    本模块不会回退到可能已过期的 JSON 信号。
    """

    # 延迟导入避免循环依赖
    from memory import TradingMemory
    from rag_tool import RAGAnalyzer

    engine = SignalFusionEngine(
        base_dir,
        allow_trade_adjustment=allow_trade_adjustment,
        api_key=api_key,
    )

    if base_signal is None:
        print("[Fusion] 必须显式传入统一策略基础信号")
        return None

    if not base_signal or base_signal.get('direction') is None:
        print("[Fusion] 信号数据不完整")
        return None

    # 加载记忆
    mem = TradingMemory(base_dir)

    # 加载 RAG (如果索引存在)
    rag = None
    try:
        rag = RAGAnalyzer(base_dir, api_key=api_key)
        # 检查索引是否已构建
        stats = rag.vector_store.get_stats() if hasattr(rag, 'vector_store') else {}
        if stats.get('total_chunks', 0) == 0:
            rag = None
    except Exception:
        rag = None

    # 提取异常因子
    factor_alerts = _extract_factor_alerts(base_dir)

    # 融合
    fused = engine.fuse(
        base_signal=base_signal,
        memory=mem,
        rag=rag,
        factor_alerts=factor_alerts,
        use_llm=use_llm,
        persist=persist,
    )

    return fused


def _extract_factor_alerts(base_dir) -> list:
    """从因子数据和特征重要性提取异常因子"""
    alerts = []
    try:
        factor_path = os.path.join(base_dir, "outputs", "df_factors.pkl")
        imp_path = os.path.join(base_dir, "outputs", "feature_importance.csv")
        if not os.path.exists(factor_path) or not os.path.exists(imp_path):
            return alerts

        df = pd.read_pickle(factor_path)
        imp = pd.read_csv(imp_path)
        if len(df) == 0 or len(imp) == 0:
            return alerts

        top_features = imp[imp['importance'] > 0].head(15)['feature'].tolist()
        latest = df.iloc[-1]

        for feat in top_features:
            if feat not in df.columns:
                continue
            series = df[feat].dropna()
            if len(series) < 100:
                continue
            cur = latest[feat]
            pct = (series < cur).mean() * 100
            if pct > 75 or pct < 25:
                direction = '偏高' if pct > 75 else '偏低'
                alerts.append((feat, direction, pct))
    except Exception:
        pass

    return alerts
