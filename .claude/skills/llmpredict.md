---
name: llmpredict
description: Run DeepSeek LLM direct prediction experiment and compare with LightGBM
---

Run the LLM direct prediction experiment: DeepSeek predicts daily TL futures direction, then compare with LightGBM daily signals.

```bash
cd "D:\桌面\F_Agent\src" && DEEPSEEK_API_KEY="YOUR_DEEPSEEK_API_KEY" python llm_predictor.py
```

This runs 93 daily predictions (~$0.10 API cost, ~1 min) and prints a comparison report.
