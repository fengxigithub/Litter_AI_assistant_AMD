#!/usr/bin/env python3
"""
文档训练集成器 - 复用现有训练框架
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def train_document_qa(data_path: str, use_existing_trainer: bool = True):
    """
    训练文档问答模型

    Args:
        data_path: 训练数据路径
        use_existing_trainer: 是否使用现有的train_manager
    """
    print("=" * 60)
    print("📚 文档问答模型训练")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"❌ 训练数据不存在: {data_path}")
        return None

    if use_existing_trainer:
        # 使用现有的训练管理器
        try:
            from src.training.train_manager import ModelTrainer

            trainer = ModelTrainer()

            print("🔧 使用现有训练框架...")

            # 训练配置（针对文档QA优化）
            config = {
                "epochs": 4,  # 文档需要更多轮次
                "batch_size": 2,
                "learning_rate": 3e-5,
                "max_length": 768,  # 文档需要更长上下文
            }

            # 开始训练
            model_path = trainer.train_full_model(
                data_path=data_path,
                config=config
            )

            if model_path:
                print(f"\n🎉 文档QA训练完成！")
                print(f"📁 模型: {model_path}")

                # 测试一下
                test_questions = [
                    "请总结文档内容",
                    "文档中的关键信息是什么？",
                    "根据文档回答具体问题"
                ]

                print("\n🧪 测试文档问答:")
                for q in test_questions:
                    print(f"  Q: {q}")
                    print(f"  A: [训练后模型会基于文档回答]")

                return model_path

        except ImportError as e:
            print(f"⚠️  无法导入现有训练器: {e}")
            print("💡 将使用简化训练...")
            use_existing_trainer = False

    if not use_existing_trainer:
        # 简化训练
        print("🔧 使用简化训练...")
        # 这里可以添加简化训练逻辑
        print("💡 建议先使用现有训练系统")
        return None


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="文档QA训练")
    parser.add_argument("--data", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--output", type=str,
                        default="./models/document_qa",
                        help="输出目录")

    args = parser.parse_args()

    # 训练
    model_path = train_document_qa(args.data)

    if model_path:
        print(f"\n✅ 训练完成！")
        print(f"📁 模型路径: {model_path}")

        # 保存配置信息
        info_file = Path(model_path) / "document_qa_info.json"
        import json
        info = {
            "model_type": "document_qa",
            "training_data": args.data,
            "training_time": "auto_generated",
            "usage": "专用于文档问答的微调模型"
        }

        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"📋 配置信息: {info_file}")


if __name__ == "__main__":
    main()