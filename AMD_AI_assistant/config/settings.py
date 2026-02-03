# 文件：settings.py
"""
配置文件管理
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.ensure_directories()

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ 加载配置文件失败: {e}")
                return self.create_default_config()
        else:
            return self.create_default_config()

    def create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        default_config = {
            "version": "1.0.0",
            "model_settings": {
                "default_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "cache_directory": "./model_cache",
                "use_mirror": True,
                "mirror_url": "https://hf-mirror.com"
            },
            "generation_settings": {
                "default_max_tokens": 200,
                "default_temperature": 0.7,
                "default_top_p": 0.9,
                "enable_history": True,
                "max_history_length": 10
            }
        }

        # 保存默认配置
        self.save_config(default_config)
        print("✅ 已创建默认配置文件")
        return default_config

    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """保存配置"""
        if config is None:
            config = self.config

        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ 配置已保存到: {self.config_file}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")

    def ensure_directories(self):
        """确保必要的目录存在"""
        dirs = [
            self.get_cache_dir(),
            Path("./logs"),
            Path("./backups"),
            Path("./exports")
        ]

        for directory in dirs:
            directory.mkdir(exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config

        # 遍历创建嵌套字典
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value
        self.save_config()

    def get_model_config(self, model_key: str) -> Dict[str, Any]:
        """获取特定模型的配置"""
        models = self.get("model_settings.available_models", {})
        return models.get(model_key, {})

    def get_cache_dir(self) -> Path:
        """获取缓存目录"""
        cache_dir = self.get("model_settings.cache_directory", "./model_cache")
        return Path(cache_dir)

    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return {
            "max_new_tokens": self.get("generation_settings.default_max_tokens", 200),
            "temperature": self.get("generation_settings.default_temperature", 0.7),
            "top_p": self.get("generation_settings.default_top_p", 0.9),
            "repetition_penalty": self.get("generation_settings.default_repetition_penalty", 1.2),
        }

    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 50)
        print("📋 当前配置摘要")
        print("=" * 50)

        print(f"版本: {self.get('version')}")
        print(f"默认模型: {self.get('model_settings.default_model')}")
        print(f"缓存目录: {self.get_cache_dir()}")
        print(f"使用镜像: {self.get('model_settings.use_mirror')}")

        gen_config = self.get_generation_config()
        print(f"生成长度: {gen_config['max_new_tokens']} tokens")
        print(f"温度: {gen_config['temperature']}")
        print(f"启用历史: {self.get('generation_settings.enable_history')}")

        print("=" * 50)


# 全局配置实例
config = ConfigManager()

if __name__ == "__main__":
    # 测试配置管理
    config.print_summary()

    # 修改配置示例
    # config.set("generation_settings.default_max_tokens", 250)
    # config.save_config()