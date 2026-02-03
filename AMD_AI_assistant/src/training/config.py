#!/usr/bin/env python3
"""
训练配置文件
"""

# 模型缓存目录
MODEL_CACHE_DIR = r"模型所在文件夹"

# 基础模型配置
BASE_MODELS = {
    "Qwen2.5-0.5B": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "cache_dir": MODEL_CACHE_DIR,
        "size_gb": 1.0,
    },
    "Qwen2.5-1.5B": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "cache_dir": MODEL_CACHE_DIR,
        "size_gb": 3.0,
    },
    "Qwen2.5-3B": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "cache_dir": MODEL_CACHE_DIR,
        "size_gb": 6.0,
    }
}