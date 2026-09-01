# 双层 LightGBM Mixture-of-Experts 设计

- **日期**: 2026-09-01
- **状态**: 已批准，待实现
- **目标**: 用可学习门控 + 软路由的 MoE 替换现有硬路由双模型，架构清晰可讲 + 严格 walk-forward 验证不倒退（遵循 CLAUDE.md rule #5）。

---

## 1. 背景与现状

现有系统已是**硬路由 2 专家**结构：
- `model_base`（在**全样本**训练）覆盖正常市（Market_Regime==0）
- `model_active`（在 regime 1/2 样本训练）覆盖高波动/趋势市
- 路由 = 手工阈值规则 `Market_Regime`（`factor_extraction.py:376-378`）：默认 0；`RV_Percentile>0.85` → 1；5-bar 趋势一致 → 2
- 路由是**硬**的（每行归一个模型），gate 是固定规则、非学习。

MoE 升级做三件事：
1. 拆成 3 个**独立**专家（正常/高波/趋势各一个，base 不再覆盖全样本）
2. 硬阈值规则 → **可学习 LightGBM 门控分类器**
3. 硬 argmax → **软加权融合** `ŷ = Σ gᵢ(x)·fᵢ(x)`

## 2. 整体架构（"双层"定义）

```
                     ┌─── 第一层: Gating Layer ───┐
市场特征 x  ──────►  │  LightGBM 门控分类器          │
                     │  输入: 波动/利差/流动性因子    │
                     │  输出: softmax(g₀,g₁,g₂)      │
                     └───────────┬───────────────────┘
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                         ▼
  ┌───────────┐          ┌───────────┐            ┌───────────┐
  │ Expert 0  │          │ Expert 1  │            │ Expert 2  │  ← 第二层
  │ 正常市    │          │ 高波动市  │            │ 趋势市    │
  └─────┬─────┘          └─────┬─────┘            └─────┬─────┘
        │ f₀(x)                │ f₁(x)                  │ f₂(x)
        └──────────────────────┼────────────────────────┘
                               ▼
            软路由: ŷ = g₀·f₀(x) + g₁·f₁(x) + g₂·f₂(x)
                               ▼
                      Pred_Ret → 甜点区排名信号
```

**双层含义**（面试可讲）：第一层门控层（学"当前信谁"），第二层专家层（各专家在自己 regime 上学预测）。定义明确，不是自创名词。

## 3. 专家训练（第二层）

三个专家各自**只在对应 regime 的样本上训练**：

| 专家 | regime | 超参（沿用现有已验证配置） |
|------|--------|------|
| Expert 0 正常 | ==0 | max_depth=3, lambda_l1/l2=15, feature_fraction=0.3, min_child_samples=350, MAE |
| Expert 1 高波动 | ==1 | max_depth=4, lambda=10, feature_fraction=0.4, min_child_samples=250, MAE |
| Expert 2 趋势 | ==2 | 同 highvol 起步 |

- **样本护栏**：某 regime 训练样本 < 1000 → 该专家回退为全样本 base 专家（防趋势市样本过少过拟合，呼应 finding #1/#8）。
- 全部沿用 12 月窗口 + 90d 时间衰减权重、RobustScaler、155 特征全集。

## 4. 门控层训练（第一层）

- **特征子集**（只判断市场状态，不重学预测）：
  `RV_Percentile, Vol_Regime, ATR_14, Vol_Surge, RV_30, RV_120, Trend_Consistency, Is_High_Vol, CN_US_10Y_Spread, Basis_ZScore_20` + 自动检测到的 Liquidity/流动性类因子（存在则纳入）。
- **标签生成**：在**验证集**样本上（非训练集，避免标签泄漏），让 3 个专家各自预测，取误差 `|fᵢ(x) − y|` 最小的专家索引作为标签。
- **模型**：多分类 LightGBM（`objective=multiclass, num_class=3`，轻正则 max_depth=3）。
- **输出**：`predict_proba` → softmax 权重 (g₀,g₁,g₂)。
- **防坍塌**：记录三专家实际主导（argmax gate）占比，若某专家 <5% 则告警（不强制均衡，仅监控）。

## 5. 融合 + 信号衔接

- `ŷ = Σ gᵢ(x)·fᵢ(x)` → 写入 `Pred_Ret`。
- **下游甜点区排名信号 / 回测 / 交易记忆完全不变**，仅 `Pred_Ret` 来源改为 MoE。
- `Model_Used` 改为记录主导专家名 + 三权重（供 Dashboard 展示）。

## 6. 验证（rule #5：不倒退才上线）

在同一 test 集（时间序列 70/15/15 切分，与现有一致）并排对比：
- 基线：现有 base/active 硬路由（Test IC 0.0388）
- MoE：软路由融合 IC

输出：整体 IC + 分 regime IC + 门控权重分布 + 主导专家占比。
**判据**：MoE 组合 IC ≥ 基线 IC 才允许替换现有 `trained_model.pkl` 使用路径；否则保留 `moe_model.pkl` 但标注实验、不改 inference 默认路径。

## 7. 落地文件

- **新增** `src/moe_model.py`：`train_experts()` + `train_gate()` + `moe_predict()` + `run_process()`（含 base/active 基线对比）。
- **修改** `src/main.py`：新增 `--mode moe`。
- **产物** `models/moe_model.pkl`：`{experts: {0,1,2}, gate, gate_features, scaler, features, regime_fallback}`。
- **修改** `src/inference.py`：`SignalGenerator` 增加 MoE 分支——若 `moe_model.pkl` 存在且验证通过则优先加载软路由，否则回退现 base/active 逻辑。
- **修改** Dashboard「🔧 微调迭代」tab：加 MoE 门控权重可视化。

## 8. 非目标（YAGNI）

- 不做无监督聚类 regime、不新增反转/流动性细分专家（样本切碎风险）。
- 不做树模型 load-balancing loss 正则项（仅监控占比）。
- 不改因子体系、不改甜点区参数。
