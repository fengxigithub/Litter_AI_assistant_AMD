"""
config.py - 项目配置文件
"""

import os
import sys
from pathlib import Path


# ==================== 根目录检测 ====================
def get_project_root():
    """
    自动检测项目根目录
    按优先级尝试多种方式：
    1. 当前文件所在目录的父级（如果是模块化结构）
    2. 环境变量指定的目录
    3. 当前工作目录
    """
    # 方法1：基于__file__的路径（最可靠）
    try:
        # 获取当前文件的绝对路径（config.py所在位置）
        current_file = Path(__file__).resolve()



        # 如果config.py在web目录下，返回上一级
        if current_file.parent.name == "web":
            return current_file.parent.parent.parent
            # 如果config.py在src目录下，返回上一级
        elif current_file.parent.name == "src":
            return current_file.parent.parent

        # 其他情况，向上找包含README.md的目录作为根目录
        for parent in current_file.parents:
            if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
                return parent
    except Exception as e:
        print(f"⚠️  基于__file__检测根目录失败: {e}")

    # 方法2：环境变量
    project_root_env = os.environ.get("AMD_AI_PROJECT_ROOT")
    if project_root_env:
        env_root = Path(project_root_env).resolve()
        if env_root.exists():
            return env_root

    # 方法3：当前工作目录
    cwd = Path.cwd()
    if (cwd / "README.md").exists() or (cwd / "requirements.txt").exists():
        return cwd

    # 方法4：脚本运行目录
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller打包后的临时目录
        bundle_dir = Path(sys._MEIPASS)
        if (bundle_dir / "README.md").exists():
            return bundle_dir

    # 最后尝试：假设当前目录是根目录
    return Path.cwd()


# 全局变量 - 项目根目录
PROJECT_ROOT = get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for directory in [MODELS_DIR, CACHE_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ==================== 模型配置 ====================
# 模型基本配置模板
MODEL_CONFIG_TEMPLATE = {
    "description": "",
    "size_gb": 0,
    "recommended_vram": 0,
    "cache_dir": None,
    "type": "unknown"
}

# HuggingFace模型配置
HF_MODELS = {
    "Qwen2.5-0.5B": {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "description": "速度快，适合聊天",
        "size_gb": 1.0,
        "recommended_vram": 4,
        "cache_dir": str(CACHE_DIR / "huggingface"),
        "type": "huggingface"
    },
    "Qwen2.5-1.5B": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "平衡性能，更聪明",
        "size_gb": 3.0,
        "recommended_vram": 8,
        "cache_dir": str(CACHE_DIR / "huggingface"),
        "type": "huggingface"
    },
    "Qwen2.5-3B": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "description": "能力强，回答详细",
        "size_gb": 6.0,
        "recommended_vram": 12,
        "cache_dir": str(CACHE_DIR / "huggingface"),
        "type": "huggingface"
    }
}

# 本地模型路径映射（相对路径）
LOCAL_MODEL_PATHS = {
    "Qwen3-0.6B": MODELS_DIR / "qianwen3" / "qianwen0.6",
    "Qwen2.5-0.5B-文档版": MODELS_DIR / "trained" / "20260129_195952" / "final_model",
    "Qwen2.5-0.5B-雪雪训练": MODELS_DIR / "trained" / "20260201_214732" / "checkpoint-400",
    "Qwen2.5-0.5B-阿米娅训练": MODELS_DIR / "trained" / "20260202_170728" / "final_model"
}

# ==================== 生成配置 ====================
GENERATION_CONFIG = {
    "default": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "pad_token_id": None,  # 自动设置
        "eos_token_id": None  # 自动设置
    },
    "creative": {
        "max_new_tokens": 768,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True
    },
    "technical": {
        "max_new_tokens": 384,
        "temperature": 0.3,
        "top_p": 0.95,
        "repetition_penalty": 1.02,
        "do_sample": False
    }
}

# ==================== 界面配置 ====================
UI_CONFIG = {
    "server_port": 7860,
    "server_name": "0.0.0.0",
    "share": False,
    "theme": "soft",
    "height": 550,
    "chatbot_height": 550,
    "memory_file": str(DATA_DIR / "conversation_memory.pkl"),
    "max_memory_items": 10
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "ai_assistant.log"),
    "max_size_mb": 10,
    "backup_count": 5
}

# ==================== 导出所有配置 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("📁 项目配置信息")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"模型目录: {MODELS_DIR}")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"日志目录: {LOGS_DIR}")
    print("=" * 60)
    print(f"可用的HuggingFace模型: {list(HF_MODELS.keys())}")
    print(f"可用的本地模型路径: {list(LOCAL_MODEL_PATHS.keys())}")
    print("=" * 60)