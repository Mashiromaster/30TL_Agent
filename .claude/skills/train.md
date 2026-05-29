---
name: train
description: Run full pipeline: factor extraction → LightGBM training → backtest
---

Run the full training pipeline and report the results.

```bash
cd "D:\桌面\F_Agent\src" && python main.py --mode train
```

After completion, summarize the key metrics: overall IC, per-regime IC, Sharpe ratio, cumulative return, and max drawdown.
