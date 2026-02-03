#!/usr/bin/env python3
"""
训练管理器 - 基于现有项目结构的训练模块
"""

import os
import sys
import json
import torch
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入配置文件
try:
    from .config import MODEL_CACHE_DIR, BASE_MODELS
except ImportError:
    # 如果配置文件不存在，使用默认值
    MODEL_CACHE_DIR = r"D:\PyCharm Community Edition 2024.1\26.1.22AMD 3.10.19\qianwenchat"


class ModelTrainer:
    """模型训练器 - 专门用于训练已加载的模型"""

    def __init__(self, model_manager=None):
        """
        初始化训练器

        Args:
            model_manager: 已有的模型管理器实例（可选）
        """
        self.model_manager = model_manager
        self.device = None
        self.training_config = self._load_default_config()

        print("=" * 60)
        print("🎯 AMD 7900XTX 模型训练器")
        print("=" * 60)

        self._setup_device()

        # 初始化数据目录
        self.data_dir = Path(self.training_config["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _convert_data_format(self, item):
        """
        转换不同格式的数据为统一格式
        Args:
            item: 原始数据项
        Returns:
            转换后的数据项列表（可能有多个对话轮次）
        """
        converted_items = []

        try:
            # 格式1: 魔搭数据集格式 (沐雪数据集)
            if 'conversation' in item:
                for conv in item['conversation']:
                    if 'human' in conv and 'assistant' in conv:
                        # 如果有system指令，可以添加到instruction中
                        instruction = conv['human']
                        if 'system' in item:
                            instruction = f"[系统指令: {item['system']}] {instruction}"

                        converted_items.append({
                            'instruction': instruction,
                            'response': conv['assistant']
                        })

            # 格式2: 原始格式 (您的示例数据格式)
            elif 'instruction' in item and 'response' in item:
                converted_items.append(item)

            # 格式3: 其他可能的格式 (根据实际情况扩展)
            # 例如: {'prompt': '...', 'completion': '...'}
            elif 'prompt' in item and 'completion' in item:
                converted_items.append({
                    'instruction': item['prompt'],
                    'response': item['completion']
                })

            # 格式4: {'input': '...', 'output': '...'}
            elif 'input' in item and 'output' in item:
                converted_items.append({
                    'instruction': item['input'],
                    'response': item['output']
                })

        except Exception as e:
            print(f"⚠️  数据格式转换失败: {e}")

        return converted_items


    def _load_default_config(self):
        """加载默认训练配置"""
        return {
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "cache_dir": MODEL_CACHE_DIR,
            "output_dir": str(project_root / "models/trained"),
            "data_dir": str(project_root / "data"),
            "epochs": 5,  # 增加训练轮数
            "batch_size": 1,  # 减小批大小（DirectML内存限制）
            "learning_rate": 2e-4,  # 调整学习率
            "max_length": 256,  # 减小序列长度
            "logging_steps": 1,  # 增加日志频率
            "save_steps": 50,
            "warmup_steps": 10,
            "gradient_accumulation_steps": 8,  # 增加梯度累积
            "fp16": False,
            "gradient_checkpointing": True,
            "lr_scheduler_type": "cosine",  # 添加学习率调度器
            "weight_decay": 0.01,  # 添加权重衰减
        }

    def _setup_device(self):
        """设置训练设备 - 优化版"""
        try:
            import torch_directml
            self.device = torch_directml.device()
            print(f"✅ 训练设备: DirectML ({self.device})")

            # DirectML特定配置
            self.training_config.update({
                "fp16": False,
                "batch_size": 1,  # DirectML通常需要较小批次
                "gradient_accumulation_steps": 8,
                "dataloader_pin_memory": False,
                "dataloader_num_workers": 0,
            })
            print("🎯 已应用DirectML优化配置")

        except ImportError:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✅ 训练设备: CUDA")
            else:
                self.device = torch.device("cpu")
                print("⚠️  训练设备: CPU（训练会很慢）")

    def prepare_training_data(self, data_type="example", data_path=None):
        """
        准备训练数据（增强版，支持魔搭数据集）

        Args:
            data_type: 数据类型 - "example"（示例）, "file"（文件）,
                      "manual"（手动输入）, "modelscope"（魔搭数据集）
            data_path: 数据文件路径（当data_type="file"时使用）

        Returns:
            data_path: 准备好的数据文件路径
        """
        print("\n📊 准备训练数据...")

        # 如果用户选择魔搭数据集
        if data_type == "modelscope":
            return self.download_modelscope_dataset()

        # 原有的数据处理逻辑
        data_dir = Path(self.training_config["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)

        if data_type == "example":
            # 创建示例数据
            data_path = data_dir / "example_training_data.jsonl"
            self._create_example_data(data_path)

        elif data_type == "file" and data_path:
            if not Path(data_path).exists():
                print(f"❌ 数据文件不存在: {data_path}")
                return None
            # 验证数据格式
            if not self._validate_data_file(data_path):
                return None

        elif data_type == "manual":
            data_path = data_dir / "manual_training_data.jsonl"
            self._collect_manual_data(data_path)

        else:
            print("❌ 不支持的数据类型")
            return None

        print(f"✅ 数据准备完成: {data_path}")
        return str(data_path)

    def _create_example_data(self, data_path, num_samples=20):
        """创建示例训练数据"""
        example_data = [
            {
                "instruction": "介绍一下Python语言",
                "response": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。它具有简洁易读的语法，广泛应用于Web开发、数据分析、人工智能等领域。"
            },
            {
                "instruction": "什么是机器学习？",
                "response": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出预测或决策，而无需显式编程。"
            },
            {
                "instruction": "如何学习编程？",
                "response": "学习编程可以从基础开始：1. 选择一门语言（如Python）2. 学习基本语法 3. 做实际项目 4. 阅读优秀代码 5. 参与开源项目。"
            },
            {
                "instruction": "AMD显卡有什么优势？",
                "response": "AMD显卡的优势包括：1. 性价比高 2. 支持开源驱动 3. 在专业计算和AI方面表现良好 4. 支持DirectML加速。"
            },
            {
                "instruction": "什么是神经网络？",
                "response": "神经网络是受生物神经网络启发的计算模型，由相互连接的节点（神经元）组成，用于模式识别和机器学习任务。"
            },
            {
                "instruction": "介绍一下Python语言的特点和应用场景",
                "response": "Python是一种高级编程语言，具有以下特点：1. 简洁易读的语法 2. 跨平台性 3. 丰富的第三方库 4. 强大的社区支持。主要应用场景包括：Web开发、数据分析、人工智能、科学计算、自动化脚本等。"
            }
        ]

        # 复制数据以达到指定数量
        all_data = []
        for i in range(num_samples):
            data = example_data[i % len(example_data)].copy()
            data["response"] = f"[样本{i + 1}] " + data["response"]
            all_data.append(data)

        # 保存为JSONL格式
        with open(data_path, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"📝 创建了 {len(all_data)} 条示例数据")

    def _validate_data_file(self, data_path):
        """验证数据文件格式"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            valid_count = 0
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    if 'instruction' in data and 'response' in data:
                        valid_count += 1
                except:
                    continue

            print(f"📊 数据文件验证: {valid_count}/{len(lines)} 条有效数据")
            return valid_count > 0

        except Exception as e:
            print(f"❌ 数据文件验证失败: {e}")
            return False

    def _collect_manual_data(self, data_path):
        """手动收集训练数据"""
        print("📝 手动输入训练数据（输入空行结束）")
        print("格式: 问题 \\n 答案")
        print("-" * 40)

        data = []
        while True:
            print(f"\n样本 #{len(data) + 1}")
            instruction = input("问题: ").strip()
            if not instruction:
                break

            response = input("答案: ").strip()
            if not response:
                print("⚠️  答案不能为空，跳过此样本")
                continue

            data.append({
                "instruction": instruction,
                "response": response
            })

            more = input("继续输入？(y/n): ").strip().lower()
            if more != 'y':
                break

        if data:
            with open(data_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✅ 保存了 {len(data)} 条训练数据")
        else:
            print("⚠️  未输入任何数据")

    def download_modelscope_dataset(self):
        """
        从魔搭社区下载数据集

        Returns:
            data_path: 下载的数据集文件路径
        """
        print("\n" + "=" * 60)
        print("🌐 魔搭社区数据集下载")
        print("=" * 60)

        # 询问用户数据集路径
        print("\n📥 请输入魔搭社区的数据集路径:")
        print("格式示例: Moemuu/Muice-Dataset")
        print("          damo/数据集名")
        print("          namespace/dataset_name")
        print("\n您可以在魔搭数据集页面找到这个路径")
        dataset_path = input("数据集路径: ").strip()

        if not dataset_path:
            print("❌ 未输入数据集路径")
            return None

        # 提取数据集名称（用于本地文件夹命名）
        if '/' in dataset_path:
            dataset_name = dataset_path.split('/')[-1]
        else:
            dataset_name = dataset_path

        # 本地保存路径
        local_dataset_dir = self.data_dir / f"modelscope_{dataset_name}"

        # 检查是否已经下载
        if local_dataset_dir.exists():
            print(f"📁 检测到已下载的数据集: {local_dataset_dir}")

            # 检查是否有训练数据文件
            train_files = list(local_dataset_dir.glob("*train*"))
            if train_files:
                print(f"✅ 使用已下载的数据集（跳过下载）")
                train_file = self._find_training_file(local_dataset_dir)
                if train_file:
                    return str(train_file)
            else:
                print("⚠️  数据集文件夹存在但没有训练文件，重新下载...")
                try:
                    shutil.rmtree(local_dataset_dir)
                except:
                    pass

        print(f"\n📥 开始下载数据集: {dataset_path}")
        print(f"📁 保存到: {local_dataset_dir}")

        try:
            # 方法1: 使用modelscope的Python API（推荐）
            try:
                from modelscope.msdatasets import MsDataset

                print("🔧 使用ModelScope API下载...")

                # 下载数据集
                dataset = MsDataset.load(
                    dataset_path,
                    subset_name=None,  # 如果有子集可以指定
                    split=None,  # 下载所有分割
                    cache_dir=str(local_dataset_dir),
                    download_mode="force_redownload"
                )

                # 确保文件夹存在
                local_dataset_dir.mkdir(parents=True, exist_ok=True)

                # 处理下载的数据集
                train_file = self._process_downloaded_dataset(dataset, local_dataset_dir)

                if train_file:
                    print(f"✅ 数据集下载完成: {train_file}")
                    return str(train_file)
                else:
                    print("❌ 无法处理下载的数据集")
                    return None

            except ImportError:
                print("⚠️  modelscope库未安装，使用命令行下载...")
                # 方法2: 使用命令行下载
                return self._download_via_commandline(dataset_path, local_dataset_dir)

        except Exception as e:
            print(f"❌ 下载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_downloaded_dataset(self, dataset, output_dir):
        """处理下载的数据集，转换为JSONL格式"""
        output_dir = Path(output_dir)

        # 如果是多文件数据集，dataset可能是字典
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                print(f"📊 处理分割: {split_name}")

                # 保存为JSONL格式
                split_file = output_dir / f"{split_name}.jsonl"
                self._save_dataset_as_jsonl(split_data, split_file)
        else:
            # 单个数据集
            train_file = output_dir / "train.jsonl"
            self._save_dataset_as_jsonl(dataset, train_file)

        # 查找训练文件
        train_file = self._find_training_file(output_dir)
        return train_file

    def _save_dataset_as_jsonl(self, dataset, output_file):
        """将数据集保存为JSONL格式"""
        print(f"💾 保存到: {output_file}")

        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            try:
                # 尝试不同的数据集访问方式
                if hasattr(dataset, '_hf_ds'):
                    # 如果是MsDataset包装的HuggingFace数据集
                    hf_ds = dataset._hf_ds
                    for item in hf_ds:
                        f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
                        count += 1
                elif hasattr(dataset, '__iter__'):
                    # 如果是可迭代对象
                    for item in dataset:
                        f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
                        count += 1
                else:
                    print(f"⚠️  未知的数据集类型: {type(dataset)}")
            except Exception as e:
                print(f"⚠️  保存数据集时出错: {e}")

        print(f"📝 保存了 {count} 条数据")

    def _download_via_commandline(self, dataset_path, local_dir):
        """通过命令行下载数据集"""
        try:
            # 确保目录存在
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            print("🔄 使用命令行下载...")

            # 构建下载命令
            cmd = [
                "modelscope",
                "download",
                "--dataset",
                dataset_path,
                "--local_dir",
                str(local_dir)
            ]

            print(f"执行命令: {' '.join(cmd)}")

            # 执行下载
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if result.returncode == 0:
                print("✅ 命令行下载成功")

                # 查找训练文件
                train_file = self._find_training_file(local_dir)
                if train_file:
                    return str(train_file)
                else:
                    print("❌ 下载成功但未找到训练文件")
                    return None
            else:
                print(f"❌ 命令行下载失败")
                print(f"错误: {result.stderr}")
                return None

        except Exception as e:
            print(f"❌ 命令行下载异常: {e}")
            return None

    def _find_training_file(self, dataset_dir):
        """在数据集目录中查找训练文件"""
        dataset_dir = Path(dataset_dir)

        if not dataset_dir.exists():
            return None

        # 优先级搜索模式
        search_patterns = [
            "train.jsonl",
            "train.json",
            "*train*.jsonl",
            "*train*.json",
            "train.csv",
            "data.jsonl",
            "dataset.jsonl",
        ]

        for pattern in search_patterns:
            files = list(dataset_dir.glob(pattern))
            if files:
                return files[0]

        # 如果没找到，返回第一个JSON/JSONL文件
        for ext in ['.jsonl', '.json', '.csv']:
            files = list(dataset_dir.glob(f"*{ext}"))
            if files:
                return files[0]

        return None

    # 以下是原有训练功能，保持不变
    def train_full_model(self, model_path=None, data_path=None, config=None, force_base_model=False): #force_base_model=False这里可选是否用基础模型，model_path=None这里是选择模型路径，文件夹即可
        """
        全参数微调训练 - 修复版
        """
        print("\n" + "=" * 60)
        print("🚀 开始全参数微调训练")
        print("=" * 60)

        # 更新配置
        if config:
            self.training_config.update(config)

        # 准备数据
        if not data_path:
            data_path = self.prepare_training_data("example")
            if not data_path:
                return None

        try:
            # 动态导入transformers
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling
            )
            from datasets import Dataset

            # 🔧 智能选择模型路径
            if force_base_model:
                load_path = self.training_config["base_model"]
                print(f"📦 强制使用基础模型: {load_path}")
            elif model_path:
                load_path = model_path
                print(f"📁 使用指定模型: {load_path}")
            else:
                # 自动查找最新的训练模型
                models_dir = Path(r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\trained")
                latest_model = None

                if models_dir.exists():
                    training_dirs = []
                    for item in models_dir.iterdir():
                        if item.is_dir() and (item / "final_model").exists():
                            training_dirs.append(item)

                    if training_dirs:
                        training_dirs.sort(key=lambda x: x.name, reverse=True)
                        latest_model = training_dirs[0] / "final_model"
                        load_path = str(latest_model)
                        print(f"✅ 自动选择最新训练模型: {latest_model.parent.name}")
                    else:
                        load_path = self.training_config["base_model"]
                        print(f"📦 使用基础模型: {load_path}")
                else:
                    load_path = self.training_config["base_model"]
                    print(f"📦 使用基础模型: {load_path}")

            # 判断是否为本地路径
            is_local_path = Path(load_path).exists()
            print(f"📊 模型类型: {'本地模型' if is_local_path else 'HuggingFace模型'}")

            # 1. 加载tokenizer
            print(f"\n🔧 加载tokenizer...")
            cache_dir = r"D:\PyCharm Community Edition 2024.1\26.1.22AMD 3.10.19\qianwenchat"

            if is_local_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    load_path,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 2. 加载模型
            print("🔧 加载模型...")
            if is_local_path:
                model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                ).to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    load_path,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True
                ).to(self.device)

            # 3. 准备数据集 - 修复版
            print("📊 准备数据集...")

            # 修改后的代码：
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())

                        # 处理魔搭数据集格式
                        if 'conversation' in item:
                            # 从conversation中提取instruction和response
                            for conv in item['conversation']:
                                if 'human' in conv and 'assistant' in conv:
                                    data.append({
                                        'instruction': conv['human'],
                                        'response': conv['assistant']
                                    })
                        # 处理原始格式
                        elif 'instruction' in item and 'response' in item:
                            data.append(item)
                    except Exception as e:
                        print(f"⚠️  解析数据行失败: {e}")
                        continue

            print(f"📈 加载 {len(data)} 条训练数据")

            # 添加数据验证
            if len(data) == 0:
                print("❌ 未找到有效训练数据，请检查数据格式")
                print("💡 数据格式应该是：")
                print("  格式1: {'instruction': '...', 'response': '...'}")
                print("  格式2: {'conversation': [{'human': '...', 'assistant': '...'}]}")
                return None

            # Qwen对话格式
            def format_example(example):
                messages = [
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['response']}
                ]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                return formatted

            formatted_texts = [format_example(item) for item in data]

            # 创建Dataset
            full_dataset = Dataset.from_dict({"text": formatted_texts})

            # 分割数据集
            split_dataset = full_dataset.train_test_split(
                test_size=0.1,  # 10%作为验证集
                shuffle=True,
                seed=42
            )

            train_raw_dataset = split_dataset["train"]
            eval_raw_dataset = split_dataset["test"]

            print(f"📊 训练集: {len(train_raw_dataset)} 条，验证集: {len(eval_raw_dataset)} 条")

            # 分词函数
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.training_config["max_length"],
                    padding=False
                )

            # 对训练集和验证集进行分词
            train_dataset = train_raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

            eval_dataset = eval_raw_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"]
            )

            # 4. 训练参数
            output_dir = Path(self.training_config["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.training_config["epochs"],
                per_device_train_batch_size=self.training_config["batch_size"],
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
                warmup_steps=self.training_config["warmup_steps"],
                logging_steps=self.training_config["logging_steps"],
                save_steps=self.training_config["save_steps"],
                evaluation_strategy="steps",
                eval_steps=50,
                learning_rate=self.training_config["learning_rate"],
                lr_scheduler_type="cosine",
                weight_decay=0.01,
                fp16=self.training_config["fp16"],
                gradient_checkpointing=self.training_config["gradient_checkpointing"],
                optim="adamw_torch",
                report_to="none",
                save_total_limit=2,
                remove_unused_columns=False,
                logging_first_step=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
            )

            # 5. 数据整理器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )

            # 6. 创建Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )

            # 7. 开始训练
            print("\n🔥 开始训练...")
            print(f"📊 训练配置:")
            print(f"  • 基础模型: {Path(load_path).name if is_local_path else load_path}")
            print(f"  • 数据量: {len(train_dataset)} 条")
            print(f"  • 训练轮数: {self.training_config['epochs']}")
            print(f"  • 批大小: {self.training_config['batch_size']}")
            print(f"  • 学习率: {self.training_config['learning_rate']}")
            print(f"  • 输出目录: {output_dir}")
            print("-" * 40)

            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time

            # 8. 保存模型
            final_model_dir = output_dir / "final_model"
            trainer.save_model(str(final_model_dir))
            tokenizer.save_pretrained(str(final_model_dir))

            print(f"\n✅ 训练完成！")
            print(f"⏱️  训练时间: {training_time:.2f} 秒")
            print(f"📁 模型保存在: {final_model_dir}")

            return str(final_model_dir)

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train_lora(self, model_path=None, data_path=None, config=None):
        """
        LoRA微调训练（显存要求低）

        Args:
            model_path: 基础模型路径
            data_path: 训练数据路径
            config: LoRA配置

        Returns:
            lora_weights_path: LoRA权重路径
        """
        print("\n" + "=" * 60)
        print("🎯 开始LoRA微调训练")
        print("=" * 60)

        try:
            # 动态导入
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )

            # 检查是否安装了peft
            try:
                from peft import LoraConfig, get_peft_model, TaskType
            except ImportError:
                print("❌ 需要安装peft库: pip install peft")
                return None

            # 默认LoRA配置
            lora_config = {
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            }

            # 更新用户配置
            if config:
                for key in ["r", "lora_alpha", "lora_dropout", "target_modules"]:
                    if key in config:
                        lora_config[key] = config[key]

            # 1. 智能选择模型路径
            if model_path:
                load_path = model_path
            else:
                models_dir = Path(r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\trained")

                latest_model = None
                if models_dir.exists():
                    training_dirs = []
                    for item in models_dir.iterdir():
                        if item.is_dir() and (item / "final_model").exists():
                            if len(item.name) == 15 and item.name[:8].isdigit():
                                training_dirs.append(item)

                    if training_dirs:
                        training_dirs.sort(key=lambda x: x.name, reverse=True)
                        latest_model = training_dirs[0] / "final_model"
                        load_path = str(latest_model)
                        print(f"✅ 使用最新训练模型: {latest_model.parent.name}")
                    else:
                        load_path = "Qwen/Qwen2.5-0.5B-Instruct"
                        print(f"📦 使用基础模型: {load_path}")
                else:
                    load_path = "Qwen/Qwen2.5-0.5B-Instruct"
                    print(f"📦 使用基础模型: {load_path}")

            # 判断是否为本地路径
            is_local_path = Path(load_path).exists()
            print(f"📊 模型类型: {'本地模型' if is_local_path else 'HuggingFace模型'}")

            # 2. 加载tokenizer
            print("🔧 加载tokenizer...")
            cache_dir = r"D:\PyCharm Community Edition 2024.1\26.1.22AMD 3.10.19\qianwenchat"

            if is_local_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    load_path,
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 3. 加载模型
            print("🔧 加载模型...")
            try:
                # 尝试量化加载
                try:
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )

                    model = AutoModelForCausalLM.from_pretrained(
                        load_path,
                        quantization_config=bnb_config,
                        trust_remote_code=True,
                        device_map="auto",
                        cache_dir=cache_dir if not is_local_path else None
                    )
                    print("✅ 使用4位量化加载")

                except Exception as e:
                    print(f"⚠️  量化加载失败，使用普通精度: {e}")
                    if is_local_path:
                        model = AutoModelForCausalLM.from_pretrained(
                            load_path,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            local_files_only=True
                        ).to(self.device)
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            load_path,
                            cache_dir=cache_dir,
                            torch_dtype=torch.float32,
                            trust_remote_code=True,
                            local_files_only=True
                        ).to(self.device)

            except Exception as e:
                print(f"❌ 加载模型失败: {e}")
                return None

            # 4. 应用LoRA
            print("🎛️  应用LoRA配置...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **lora_config
            )

            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            # 5. 准备数据
            print("📊 准备训练数据...")
            if not data_path:
                print("❌ 未提供训练数据")
                return None

            dataset = self._prepare_dataset(tokenizer, data_path)

            # 6. 训练参数
            output_dir = Path(self.training_config["output_dir"]) / f"lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=config.get("epochs", 3) if config else 3,
                per_device_train_batch_size=config.get("batch_size", 4) if config else 4,
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4) if config else 4,
                warmup_steps=config.get("warmup_steps", 50) if config else 50,
                logging_steps=config.get("logging_steps", 10) if config else 10,
                save_steps=config.get("save_steps", 50) if config else 50,
                learning_rate=config.get("learning_rate", 2e-4) if config else 2e-4,
                fp16=self.training_config.get("fp16", False),
                gradient_checkpointing=True,
                optim="adamw_torch",
                report_to="none",
                save_total_limit=2,
                remove_unused_columns=False,
            )

            # 7. 数据整理器
            # 修改DataCollator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # 添加这个，可能有助于DirectML性能
            )

            # 8. 创建Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            # 9. 开始训练
            print("\n🔥 开始LoRA训练...")
            print(f"📊 训练配置:")
            print(f"  • 基础模型: {Path(load_path).name if is_local_path else load_path}")
            print(f"  • 数据量: {len(dataset)} 条")
            print(f"  • 训练轮数: {training_args.num_train_epochs}")
            print(f"  • 批大小: {training_args.per_device_train_batch_size}")
            print(f"  • 学习率: {training_args.learning_rate}")
            print(f"  • LoRA配置: r={lora_config['r']}, alpha={lora_config['lora_alpha']}")
            print(f"  • 输出目录: {output_dir}")
            print("-" * 40)

            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time

            # 10. 保存LoRA权重
            lora_dir = output_dir / "lora_weights"
            model.save_pretrained(str(lora_dir))
            tokenizer.save_pretrained(str(lora_dir))

            print(f"\n✅ LoRA训练完成！")
            print(f"⏱️  训练时间: {training_time:.2f} 秒")
            print(f"📁 LoRA权重保存在: {lora_dir}")

            return str(lora_dir)

        except Exception as e:
            print(f"❌ LoRA训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_dataset(self, tokenizer, data_path):
        """准备数据集 - 修复版本"""
        # 加载数据
        # 修改后的代码：
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())

                    # 处理魔搭数据集格式
                    if 'conversation' in item:
                        # 从conversation中提取instruction和response
                        for conv in item['conversation']:
                            if 'human' in conv and 'assistant' in conv:
                                data.append({
                                    'instruction': conv['human'],
                                    'response': conv['assistant']
                                })
                    # 处理原始格式
                    elif 'instruction' in item and 'response' in item:
                        data.append(item)
                except Exception as e:
                    print(f"⚠️  解析数据行失败: {e}")
                    continue

        print(f"📈 加载 {len(data)} 条训练数据")

        # 添加数据验证
        if len(data) == 0:
            print("❌ 未找到有效训练数据，请检查数据格式")
            # 返回一个空的Dataset或者None
            from datasets import Dataset
            return Dataset.from_dict({"text": []})

        # Qwen对话格式 - 更精确的格式
        def format_example(example):
            messages = [
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example['response']}
            ]
            # 使用tokenizer的apply_chat_template方法
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            return formatted

        formatted_texts = [format_example(item) for item in data]

        # 创建Dataset
        from datasets import Dataset
        dataset = Dataset.from_dict({"text": formatted_texts})

        # 分词函数 - 修复版本
        def tokenize_function(examples):
            # 只对文本进行分词，不添加特殊token
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.training_config["max_length"],
                padding=False  # 改为动态padding
            )

            # 手动添加标签（用于因果语言建模）
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # 应用分词
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        return tokenized_dataset

    def test_trained_model(self, model_path, test_prompts=None):
        """测试训练后的模型"""
        print("\n🧪 测试训练后的模型...")

        if test_prompts is None:
            test_prompts = [
                "介绍一下Python语言",
                "什么是机器学习？",
                "AMD显卡有什么优势？"
            ]

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # 加载训练后的模型
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)

            model.eval()

            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n📝 测试 {i}: {prompt}")

                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(text, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    response = tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True
                    )

                print(f"🤖 回复: {response}")

        except Exception as e:
            print(f"❌ 测试失败: {e}")


def main():
    """训练脚本主入口"""
    trainer = ModelTrainer()

    print("\n请选择训练模式:")
    print("1. 全参数微调")
    print("2. LoRA微调（推荐，显存要求低）")
    print("3. 准备训练数据")
    print("4. 测试训练后的模型")
    print("5. 🌐 下载魔搭社区数据集")

    choice = input("\n请输入选择 (1-5): ").strip()

    if choice == "1":
        # 全参数微调
        config = {
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 5e-5,
        }

        data_type = input("数据类型 (1=示例, 2=文件, 3=手动输入, 4=魔搭数据集): ").strip()
        if data_type == "1":
            data_path = None  # 使用示例数据
        elif data_type == "2":
            data_path = input("数据文件路径: ").strip()
        elif data_type == "3":
            data_path = trainer.prepare_training_data("manual")
        elif data_type == "4":
            data_path = trainer.prepare_training_data("modelscope")
        else:
            print("❌ 无效选择")
            return

        model_path = trainer.train_full_model(data_path=data_path, config=config)

        if model_path:
            test = input("是否测试训练后的模型？(y/n): ").strip().lower()
            if test == 'y':
                trainer.test_trained_model(model_path)

    elif choice == "2":
        # LoRA微调
        config = {
            "r": 8,
            "lora_alpha": 32,
            "epochs": 3,
            "batch_size": 4,
        }

        data_type = input("数据类型 (1=示例, 2=文件, 3=魔搭数据集): ").strip()
        if data_type == "1":
            data_path = None
        elif data_type == "2":
            data_path = input("数据文件路径: ").strip()
        elif data_type == "3":
            data_path = trainer.prepare_training_data("modelscope")
        else:
            print("❌ 无效选择")
            return

        lora_path = trainer.train_lora(data_path=data_path, config=config)

        if lora_path:
            print(f"\n💡 LoRA权重路径: {lora_path}")
            print("💡 要在推理代码中使用LoRA，需要:")
            print("  1. 加载基础模型")
            print("  2. 使用PeftModel加载LoRA权重")
            print("  3. 使用model.merge_and_unload()合并权重")

    elif choice == "3":
        # 准备训练数据
        print("\n准备训练数据:")
        print("1. 创建示例数据")
        print("2. 手动输入数据")

        sub_choice = input("请选择: ").strip()
        if sub_choice == "1":
            num_samples = input("样本数量 (默认20): ").strip()
            num_samples = int(num_samples) if num_samples else 20
            trainer.prepare_training_data("example")
        elif sub_choice == "2":
            trainer.prepare_training_data("manual")

    elif choice == "4":
        # 测试训练后的模型
        model_path = input("训练模型路径: ").strip()
        if os.path.exists(model_path):
            trainer.test_trained_model(model_path)
        else:
            print("❌ 模型路径不存在")

    elif choice == "5":
        # 下载魔搭社区数据集
        data_path = trainer.prepare_training_data("modelscope")
        if data_path:
            print(f"\n✅ 数据集已下载到: {data_path}")
            use_now = input("是否立即使用此数据集进行训练？(y/n): ").strip().lower()
            if use_now == 'y':
                # 询问训练类型
                print("\n选择训练类型:")
                print("1. 全参数微调")
                print("2. LoRA微调")

                train_choice = input("请选择 (1/2): ").strip()

                if train_choice == "1":
                    config = {
                        "epochs": 3,
                        "batch_size": 2,
                        "learning_rate": 5e-5,
                    }
                    trainer.train_full_model(data_path=data_path, config=config)
                elif train_choice == "2":
                    config = {
                        "r": 8,
                        "lora_alpha": 32,
                        "epochs": 3,
                        "batch_size": 4,
                    }
                    trainer.train_lora(data_path=data_path, config=config)
                else:
                    print("❌ 无效选择")

    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    # 检查modelscope是否已安装
    try:
        import modelscope

        print("✅ modelscope库已安装")
    except ImportError:
        print("⚠️  modelscope库未安装")
        install = input("是否现在安装？(y/n): ").strip().lower()
        if install == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
                print("✅ modelscope安装成功")
            except:
                print("❌ 安装失败，部分功能可能无法使用")

    main()