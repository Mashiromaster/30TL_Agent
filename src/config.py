# -*- coding: utf-8 -*-
"""项目路径配置 — 自动探测项目根,跨平台通用。

各模块以 `from config import BASE_DIR` 引入,替代硬编码的绝对路径。
BASE_DIR 导出为 str,与现有 os.path.join(base_dir, ...) 用法完全兼容。
"""
from pathlib import Path

# 项目根 = 本文件(src/config.py)的上一级目录
BASE_DIR = str(Path(__file__).resolve().parent.parent)
