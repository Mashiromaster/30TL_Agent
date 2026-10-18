# -*- coding: utf-8 -*-
"""Factor formula registry, transcribed from the computing source.

Every entry records where the factor is actually computed, so a formula shown
in the UI can be checked against the code. ``status`` distinguishes factors the
current pipeline still computes from names that survive only in stored data:
a retired factor's formula describes how the existing column was produced, not
what a rebuild would produce.

Keep ``source`` citations accurate when the computation moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field


LIVE = "live"
RETIRED = "retired"


@dataclass(frozen=True)
class FactorFormula:
    name: str
    label: str
    category: str
    formula: str
    inputs: tuple[str, ...]
    window: str
    normalization: str
    source: str
    status: str = LIVE
    notes: tuple[str, ...] = field(default_factory=tuple)
    replacement: str | None = None


def _f(**kwargs):
    return FactorFormula(**kwargs)


FORMULAS: dict[str, FactorFormula] = {entry.name: entry for entry in (
    _f(
        name="Macro_Surprise_Composite",
        label="宏观意外综合",
        category="宏观",
        formula="mean(PMI_ZScore, M2_Surprise, SocialFin_ZScore)  # 按行取可用列均值",
        inputs=("PMI_ZScore", "M2_Surprise", "SocialFin_ZScore"),
        window="取决于各分量（PMI 为约 24 个月滚动 Z 分数）",
        normalization="分量各自 Z 标准化后直接平均，无裁剪",
        source="src/macro_factors.py（f44e254 之前的版本）",
        status=RETIRED,
        replacement="Macro_State_Composite",
        notes=(
            "当前代码已不再计算该因子，commit f44e254 删除。",
            "现有列来自删除前生成的 df_factors.pkl，重新构建因子不会再产生它。",
            "替代因子 Macro_State_Composite 成员不同且先裁剪到 ±4 再平均，两者不等价。",
        ),
    ),
    _f(
        name="PMI_ZScore",
        label="PMI 制造业 Z 分数",
        category="宏观",
        formula="clip((PMI - rolling_mean(PMI, 720)) / (rolling_std(PMI, 720) + 0.01), -4, 4)",
        inputs=("PMI_Manufacturing",),
        window="720 行日频（月度值前向填充后约 24 个月），min_periods=60",
        normalization="滚动 Z 分数，分母加 0.01，裁剪到 ±4",
        source="src/macro_factors.py（f44e254 之前的版本）",
        status=RETIRED,
        replacement="PMI_State_ZScore",
        notes=(
            "当前代码已不再计算该因子，commit f44e254 删除。",
            "替代因子 PMI_State_ZScore 用 24 个月度观测而非 720 个填充日频行，且不加 epsilon、不裁剪。",
        ),
    ),
    _f(
        name="Macro_State_Composite",
        label="宏观状态综合",
        category="宏观",
        formula="mean(clip([PMI_State_ZScore, CPI_Acceleration_3M, M2_State_ZScore, SocialFin_YoY_ZScore], -4, 4), skipna=True)",
        inputs=("PMI_State_ZScore", "CPI_Acceleration_3M",
                "M2_State_ZScore", "SocialFin_YoY_ZScore"),
        window="取决于各分量（PMI/M2 为月度滚动 Z 分数）",
        normalization="各分量先裁剪到 ±4 再按行平均，缺失分量跳过",
        source="src/macro_factors.py:385-398",
        notes=(
            "Macro_Surprise_Composite 的替代因子，但成员不同且先裁剪再平均，两者不等价。",
            "只统计当前存在的分量列，成员缺失时不报错，均值口径会随之变化。",
        ),
    ),
    _f(
        name="PMI_State_ZScore",
        label="PMI 制造业状态 Z 分数",
        category="宏观",
        formula="(PMI - rolling_mean(PMI, 24)) / rolling_std(PMI, 24)",
        inputs=("PMI_Manufacturing",),
        window="24 个月度观测，min_periods=12",
        normalization="滚动 Z 分数；标准差为 0 时置 NaN。无 epsilon、无裁剪",
        source="src/macro_factors.py:326-331",
        notes=(
            "PMI_ZScore 的替代因子。窗口为 24 个月度观测，而非旧版 720 个前向填充日频行。",
            "原始数据来自 ak.macro_china_pmi 的「制造业-指数」，TTL 50 天。",
        ),
    ),
    _f(
        name="PMI_Change_3M",
        label="PMI 三个月变动",
        category="宏观",
        formula="PMI_Manufacturing.diff(3)",
        inputs=("PMI_Manufacturing",),
        window="3 个月度观测（差分）",
        normalization="无",
        source="src/macro_factors.py:332",
    ),
    _f(
        name="CPI_Acceleration_3M",
        label="CPI 三个月加速度",
        category="宏观",
        formula="CPI - CPI.shift(3)",
        inputs=("CPI_YoY",),
        window="3 个月度观测",
        normalization="无",
        source="src/macro_factors.py:344",
        notes=("CPI_Momentum 的替代因子。",),
    ),
    _f(
        name="M2_State_ZScore",
        label="M2 状态 Z 分数",
        category="宏观",
        formula="(M2 - rolling_mean(M2, 12)) / rolling_std(M2, 12)",
        inputs=("M2_YoY",),
        window="12 个月度观测",
        normalization="滚动 Z 分数；标准差为 0 时置 NaN",
        source="src/macro_factors.py:358",
        notes=("M2_Surprise 的替代因子。",),
    ),
    _f(
        name="YC_Level_10Y",
        label="10 年期收益率水平",
        category="宏观",
        formula="Yield_10Y",
        inputs=("Yield_10Y",),
        window="无（当期水平值）",
        normalization="无。单位为收益率百分点",
        source="src/macro_factors.py:195",
        notes=("YC_Level_Shift 的替代因子（旧名已不再计算）。",),
    ),
    _f(
        name="YC_Level_ZScore",
        label="10 年期收益率 Z 分数",
        category="宏观",
        formula="(Yield_10Y - rolling_mean(·, 60)) / rolling_std(·, 60)",
        inputs=("Yield_10Y",),
        window="60 个交易日观测",
        normalization="滚动 Z 分数",
        source="src/macro_factors.py:200-203",
    ),
    _f(
        name="CN_US_10Y_Spread",
        label="中美 10 年期利差",
        category="宏观",
        formula="CN_10Y - US_10Y",
        inputs=("CN_10Y", "US_10Y"),
        window="无（当期水平值）",
        normalization="无。单位为收益率百分点",
        source="src/data_fetcher.py:470 构建；src/macro_factors.py:263-266 透传",
        notes=(
            "原始数据来自 ak.bond_zh_us_rate，按中文列名匹配「中国」「10年」与「美国」「10年」。",
            "TTL 5 天，超期置 NaN 后补 0，因此 0 既可能是真实值也可能代表缺失。",
        ),
    ),
    _f(
        name="CN_US_10Y_Spread_Z",
        label="中美利差 Z 分数",
        category="宏观",
        formula="(spread - rolling_mean(spread, 60)) / rolling_std(spread, 60)",
        inputs=("CN_US_10Y_Spread",),
        window="60 个交易日观测，min_periods=20",
        normalization="滚动 Z 分数；标准差为 0 时置 NaN（不产生 inf），合并时补 0",
        source="src/macro_factors.py:267-271",
    ),
    _f(
        name="YC_Slope_30Y_10Y",
        label="30Y-10Y 期限斜率",
        category="宏观",
        formula="Yield_30Y - Yield_10Y",
        inputs=("Yield_30Y", "Yield_10Y"),
        window="无（当期水平值）",
        normalization="无。单位为收益率百分点",
        source="src/macro_factors.py:188-189",
        notes=("收益率来自 ak.bond_china_yield 的「中债国债收益率曲线」。",),
    ),
    _f(
        name="YC_Slope_10Y_1Y",
        label="10Y-1Y 期限斜率",
        category="宏观",
        formula="Yield_10Y - Yield_1Y",
        inputs=("Yield_10Y", "Yield_1Y"),
        window="无（当期水平值）",
        normalization="无。单位为收益率百分点",
        source="src/macro_factors.py:186-187",
    ),
    _f(
        name="YC_Curvature",
        label="收益率曲线曲率（蝶式）",
        category="宏观",
        formula="2 × Yield_5Y - Yield_1Y - Yield_10Y",
        inputs=("Yield_5Y", "Yield_1Y", "Yield_10Y"),
        window="无（当期水平值）",
        normalization="无。标准 1Y/5Y/10Y 蝶式，权重 (-1, +2, -1)",
        source="src/macro_factors.py:190-192",
    ),
    _f(
        name="YC_Momentum_5D",
        label="10Y 收益率 5 日变动",
        category="宏观",
        formula="Yield_10Y.diff(5)",
        inputs=("Yield_10Y",),
        window="5 个交易日（差分，非涨跌幅）",
        normalization="无",
        source="src/macro_factors.py:304-313",
        notes=(
            "Risk_On_Off = sign(该因子)，二者按构造共线。",
            "YC_10Y_Change_5D（macro_factors.py:197）是同一数值的另一个名字，两列都会进入最终因子集。",
        ),
    ),
    _f(
        name="Short_Momentum_5D",
        label="短期动量（5 日）",
        category="动量",
        formula="adj_close.shift(1).pct_change(1200)",
        inputs=("adj_close",),
        window="1200 根分钟 bar = 5 × 240",
        normalization="无。为小数收益率，非百分数",
        source="src/factor_extraction.py:403",
        notes=(
            "价格优先取 adj_close，缺失时退回 close；先 shift(1) 避免前视。",
            "未按 ticker 分组，主力合约拼接处会跨越换月边界。",
        ),
    ),
    _f(
        name="Mid_Momentum_1M",
        label="中期动量（约 1 个月）",
        category="动量",
        formula="adj_close.shift(1).pct_change(5000)",
        inputs=("adj_close",),
        window="5000 根分钟 bar ≈ 20.8 个交易日",
        normalization="无。为小数收益率",
        source="src/factor_extraction.py:404",
        notes=("未按 ticker 分组，拼接处会跨越换月边界。",),
    ),
    _f(
        name="Mid_Momentum_2M",
        label="中期动量（约 2 个月）",
        category="动量",
        formula="adj_close.shift(1).pct_change(10000)",
        inputs=("adj_close",),
        window="10000 根分钟 bar ≈ 41.7 个交易日",
        normalization="无。为小数收益率",
        source="src/factor_extraction.py:405",
        notes=("未按 ticker 分组，拼接处会跨越换月边界。",),
    ),
    _f(
        name="Basis_ZScore_20",
        label="净基差 Z 分数（20 日）",
        category="基差",
        formula="clip((NetBasis_CTD - rolling_mean(·, 20)) / (rolling_std(·, 20) + 1e-6), -3, 3)",
        inputs=("NetBasis_CTD",),
        window="20 个日频行，按 ticker 分组，min_periods=5",
        normalization="滚动 Z 分数，分母加 1e-6，裁剪到 ±3",
        source="src/factor_extraction.py:92-99",
        notes=(
            "CTD 为每个 (date, ticker) 下净基差最小的合约。",
            "可用时点延后 1 天（date + 1），再按 ticker 前向填充并补 0。",
        ),
    ),
    _f(
        name="Basis_ZScore_10",
        label="净基差 Z 分数（10 日）",
        category="基差",
        formula="clip((NetBasis_CTD - rolling_mean(·, 10)) / (rolling_std(·, 10) + 1e-6), -3, 3)",
        inputs=("NetBasis_CTD",),
        window="10 个日频行，按 ticker 分组，min_periods=5",
        normalization="滚动 Z 分数，分母加 1e-6，裁剪到 ±3",
        source="src/factor_extraction.py:101-103",
    ),
    _f(
        name="OI_Growth_5D",
        label="持仓量 5 日增速",
        category="量价",
        formula="clip(oi.shift(1).pct_change(1200), -0.2, 0.2)",
        inputs=("oi",),
        window="1200 根分钟 bar = 5 × 240",
        normalization="裁剪到 ±0.2",
        source="src/enhanced_factors.py:102",
        notes=("未按 ticker 分组。",),
    ),
    _f(
        name="OI_Growth_20D",
        label="持仓量 20 日增速",
        category="量价",
        formula="clip(oi.shift(1).pct_change(4800), -0.3, 0.3)",
        inputs=("oi",),
        window="4800 根分钟 bar = 20 × 240",
        normalization="裁剪到 ±0.3（与 5D 的 ±0.2 不同）",
        source="src/enhanced_factors.py:103",
        notes=("未按 ticker 分组。",),
    ),
)}


PIPELINE_NOTE = (
    "所有因子最后统一经过 src/factor_extraction.py:558-566："
    "inf 置 NaN、丢弃前 500 行、按 ticker 前向填充、再补 0。"
    "宏观因子另有 TTL 判定，超期先置 NaN 再补 0 —— 因此 0 无法区分「真实为零」与「缺失或过期」。"
)


def lookup(name):
    """Return the formula entry for ``name``, or None when unregistered."""
    return FORMULAS.get(str(name))
