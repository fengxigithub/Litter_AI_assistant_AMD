"""
文档问答提取器 - 优化版本
支持MacBERT/BERT-wwm/ERNIE，改进问题识别准确性
"""
import os
import json
import torch
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import time
import numpy as np

# # 设置环境变量（必须在import transformers之前）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer警告
# import os

# 强制使用国内镜像站（添加这三行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/huggingface_cache"  # 可自定义缓存路径
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"


print("=" * 60)
print("📄 文档问答提取器 - 优化版本初始化")
print("=" * 60)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForQuestionAnswering,
        pipeline,
        QuestionAnsweringPipeline
    )
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers库可用")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"❌ Transformers库导入失败: {e}")


class TextPreprocessor:
    """文本预处理器"""

    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本"""
        # 移除多余空格和换行符
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        # 移除特殊字符但保留标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：、（）《》「」【】]', ' ', text)
        return text.strip()

    @staticmethod
    def split_long_text(text: str, max_length: int = 500) -> List[str]:
        """将长文本分割成小块"""
        if len(text) <= max_length:
            return [text]

        # 尝试按标点分割
        sentences = re.split(r'([。！？；])', text)
        chunks = []
        current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

                # 如果单个句子就超过max_length，强制分割
                if len(current_chunk) > max_length:
                    # 按字符分割
                    for j in range(0, len(current_chunk), max_length):
                        chunks.append(current_chunk[j:j+max_length])
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    @staticmethod
    def normalize_question(question: str) -> str:
        """标准化问题"""
        question = question.strip()
        # 移除问题末尾标点
        question = re.sub(r'[。！？；：，、]$', '', question)
        # 确保问题以问号结尾（如果不是陈述句）
        if not question.endswith('？') and not question.endswith('?') and not question.endswith('。'):
            question += '？'
        return question


class DocumentQA:
    """优化版单模型文档问答提取器"""

    def __init__(self, model_name_or_path: str, device: str = None):
        """
        初始化文档问答模型

        Args:
            model_name_or_path: 模型名称或本地路径
            device: 运行设备 ('cuda', 'cpu', 或 None自动选择)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请先安装transformers库: pip install transformers")

        self.model_name = model_name_or_path
        self.device = device if device else self._auto_select_device()
        self.preprocessor = TextPreprocessor()

        print(f"🔧 初始化QA模型: {model_name_or_path}")
        print(f"🎮 使用设备: {self.device}")

        # 记录加载时间
        start_time = time.time()

        try:
            # 加载tokenizer和模型
            print("📥 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            )

            print("📥 加载模型...")
            # 尝试使用安全的safetensors格式加载
            try:
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    use_safetensors=True  # 优先使用safetensors
                )
            except:
                # 如果safetensors失败，回退到普通方式
                print("⚠️  safetensors加载失败，尝试普通加载...")
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True
                )

            # 创建pipeline，设置更详细的参数
            print("🔗 创建问答pipeline...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                batch_size=1
            )

            # 获取模型的最大长度
            self.max_length = self.tokenizer.model_max_length
            if self.max_length > 512:  # 限制最大长度
                self.max_length = 512

            load_time = time.time() - start_time
            print(f"✅ 模型加载完成! 耗时: {load_time:.2f}秒")
            print(f"📏 最大序列长度: {self.max_length}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 尝试备用加载方式
            try:
                print("🔄 尝试备用加载方式...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    use_fast=True
                )
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name_or_path
                ).to(self.device)

                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                print("✅ 备用加载方式成功!")
            except Exception as e2:
                raise RuntimeError(f"所有加载方式都失败: {e2}")

    def _auto_select_device(self) -> str:
        """自动选择设备"""
        # 强制优先使用GPU
        try:
            # 优先尝试DirectML（AMD显卡）
            import torch_directml
            dml_device = torch_directml.device()
            print(f"✅ 检测到AMD显卡，使用DirectML加速")
            return dml_device
        except ImportError:
            print("⚠️  未安装torch_directml，尝试使用CUDA")

        # 其次尝试CUDA（NVIDIA显卡）
        if torch.cuda.is_available():
            print("✅ 检测到NVIDIA显卡，使用CUDA加速")
            return "cuda"

        # 最后使用CPU
        print("⚠️  未检测到GPU，使用CPU")
        return "cpu"
    # def _auto_select_device(self) -> str:
    #     """自动选择设备"""
    #     if torch.cuda.is_available():
    #         return "cuda"
    #     elif hasattr(torch, 'directml'):  # 检查DirectML支持
    #         try:
    #             import torch_directml
    #             return torch_directml.device()
    #         except:
    #             return "cpu"
    #     else:
    #         return "cpu"

    def extract_answer(
        self,
        context: str,
        question: str,
        max_answer_length: int = 150,  # 增加默认长度
        top_k: int = 5,  # 增加默认top_k
        confidence_threshold: float = 0.1,  # 置信度阈值
        handle_long_document: bool = True
    ) -> List[Dict]:
        """
        从文档中提取答案（优化版）

        Args:
            context: 文档内容
            question: 问题
            max_answer_length: 最大答案长度
            top_k: 返回前k个最可能的答案
            confidence_threshold: 置信度阈值
            handle_long_document: 是否处理长文档

        Returns:
            答案列表，每个答案包含: text, score, start, end
        """
        try:
            # 预处理文本
            context = self.preprocessor.clean_text(context)
            question = self.preprocessor.normalize_question(question)

            print(f"📊 预处理后: 文档长度={len(context)}, 问题='{question}'")

            # 处理长文档
            answers = []
            if handle_long_document and len(context) > 800:
                print("📝 文档较长，启用分块处理...")
                chunks = self.preprocessor.split_long_text(context, max_length=800)

                for chunk_idx, chunk in enumerate(chunks):
                    if len(chunk) < 20:  # 跳过太短的块
                        continue

                    print(f"  处理分块 {chunk_idx+1}/{len(chunks)} (长度: {len(chunk)})")

                    chunk_answers = self._extract_from_chunk(
                        chunk, question, max_answer_length, top_k
                    )

                    for ans in chunk_answers:
                        # 调整答案位置并添加到结果
                        if ans.get("score", 0) > confidence_threshold:
                            answers.append(ans)

                # 合并相似的答案
                answers = self._merge_similar_answers(answers)
            else:
                # 直接提取
                answers = self._extract_from_chunk(
                    context, question, max_answer_length, top_k
                )

            # 过滤和排序
            filtered_answers = []
            for result in answers:
                if result.get("score", 0) > confidence_threshold:
                    filtered_answers.append(result)

            # 按置信度排序
            filtered_answers.sort(key=lambda x: x.get("score", 0), reverse=True)

            # 限制数量
            filtered_answers = filtered_answers[:top_k]

            print(f"✅ 找到 {len(filtered_answers)} 个有效答案")
            return filtered_answers

        except Exception as e:
            print(f"❌ 答案提取失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_from_chunk(
        self,
        context: str,
        question: str,
        max_answer_length: int,
        top_k: int
    ) -> List[Dict]:
        """从单个文档块提取答案"""
        try:
            # 使用pipeline提取答案
            results = self.qa_pipeline(
                {
                    "context": context,
                    "question": question
                },
                top_k=top_k,
                max_answer_len=max_answer_length,
                handle_impossible_answer=False,  # 不处理无法回答的情况
                max_seq_len=self.max_length,
                doc_stride=128  # 增加步长
            )

            # 确保结果总是列表
            if not isinstance(results, list):
                results = [results]

            # 格式化结果
            formatted_results = []
            for result in results:
                if result.get("answer"):
                    # 计算置信度（转换为百分比）
                    score = result.get("score", 0)

                    # 后处理答案文本
                    answer_text = self._postprocess_answer(result["answer"])

                    if answer_text:  # 确保答案非空
                        formatted_results.append({
                            "text": answer_text,
                            "score": round(score, 4),
                            "confidence": round(score * 100, 2),
                            "start": result.get("start", 0),
                            "end": result.get("end", 0),
                            "context": context  # 保留上下文用于调试
                        })

            return formatted_results

        except Exception as e:
            print(f"❌ 分块提取失败: {e}")
            return []

    def _postprocess_answer(self, answer: str) -> str:
        """后处理答案文本"""
        if not answer:
            return ""

        # 清理答案
        answer = answer.strip()
        answer = re.sub(r'^\s*[.,，。!！?？;；:：]\s*', '', answer)
        answer = re.sub(r'\s+', ' ', answer)

        # 移除不完整的句子
        if len(answer) < 2:  # 太短
            return ""

        return answer

    def _merge_similar_answers(self, answers: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """合并相似的答案"""
        if not answers:
            return []

        merged = []
        used = [False] * len(answers)

        for i in range(len(answers)):
            if used[i]:
                continue

            current = answers[i]
            similar_group = [current]
            used[i] = True

            # 寻找相似答案
            for j in range(i + 1, len(answers)):
                if not used[j]:
                    # 简单的文本相似度计算（基于重叠）
                    text1 = current["text"]
                    text2 = answers[j]["text"]

                    # 计算Jaccard相似度
                    set1 = set(text1)
                    set2 = set(text2)
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)

                    if union > 0 and intersection / union > similarity_threshold:
                        similar_group.append(answers[j])
                        used[j] = True

            # 合并相似答案（取置信度最高的）
            if similar_group:
                best_answer = max(similar_group, key=lambda x: x.get("score", 0))
                merged.append(best_answer)

        return merged

    def batch_extract(self, contexts: List[str], questions: List[str],
                      **kwargs) -> List[List[Dict]]:
        """批量提取答案"""
        results = []
        for context, question in zip(contexts, questions):
            answers = self.extract_answer(context, question, **kwargs)
            results.append(answers)
        return results

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "tokenizer_vocab_size": self.tokenizer.vocab_size,
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }


class QAExtractorManager:
    """QA模型管理器（多模型支持）"""

    # 预定义的模型配置 - 使用CMRC2018微调模型
    QA_MODEL_CONFIGS = {
         # ==== 问答专用模型（强烈推荐）====
        "uer-roberta-qa": {
            "name": "uer/roberta-base-chinese-extractive-qa",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\roberta-base-chinese-extractive-qa",
            "description": "UER RoBERTa抽取式问答模型（问答专用，强烈推荐）",
            "max_context_length": 512,
            "recommended_top_k": 3,
            "confidence_threshold": 0.1
        },
        "macbert-cmrc": {
            "name": "hfl/chinese-macbert-base-cmrc2018",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\macbert-cmrc",
            "description": "MacBERT在CMRC2018上微调的问答模型（推荐）",
            "max_context_length": 512,
            "recommended_top_k": 3,
            "confidence_threshold": 0.1
        },
        "bert-wwm-cmrc": {
            "name": "hfl/chinese-bert-wwm-ext-cmrc2018",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\bert-wwm-cmrc",
            "description": "BERT-wwm在CMRC2018上微调的问答模型",
            "max_context_length": 512,
            "recommended_top_k": 3,
            "confidence_threshold": 0.1
        },
        "roberta-cmrc": {
            "name": "hfl/chinese-roberta-wwm-ext-cmrc2018",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\roberta-cmrc",
            "description": "RoBERTa在CMRC2018上微调的问答模型",
            "max_context_length": 512,
            "recommended_top_k": 3,
            "confidence_threshold": 0.1
        },
        "macbert-base": {
            "name": "hfl/chinese-macbert-base",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\macbert",
            "description": "基础MacBERT模型（通用）",
            "max_context_length": 512,
            "recommended_top_k": 5,
            "confidence_threshold": 0.05  # 降低阈值
        },
        "bert-wwm-base": {
            "name": "hfl/chinese-bert-wwm-ext",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\bert-wwm",
            "description": "基础BERT-wwm模型",
            "max_context_length": 512,
            "recommended_top_k": 5,
            "confidence_threshold": 0.05
        },
        "ernie-base": {
            "name": "nghuyong/ernie-3.0-base-zh",
            "local_path": r"F:\py_work\AMD_AI_Project\AMD_AI_Project\models\qa_models\ernie",
            "description": "ERNIE 3.0基础模型",
            "max_context_length": 512,
            "recommended_top_k": 5,
            "confidence_threshold": 0.05
        }
    }

    def __init__(self):
        """初始化模型管理器"""
        self.models = {}  # 已加载的模型字典
        self.current_model = None
        self.current_model_key = None
        self.default_params = {
            "top_k": 3,
            "max_answer_length": 150,
            "confidence_threshold": 0.1,
            "handle_long_document": True
        }

        # 自动发现微调模型
        self.finetuned_models = self._discover_finetuned_models()

        print("✅ QA模型管理器初始化完成")

        # 合并预定义模型和微调模型
        all_models = list(self.QA_MODEL_CONFIGS.keys()) + list(self.finetuned_models.keys())
        print(f"📊 支持 {len(all_models)} 个模型:")
        for model in all_models:
            print(f"   • {model}")

        # 将uer-roberta-qa设为推荐模型
        self.recommended_model = "uer-roberta-qa"
        print(f"🌟 推荐模型: {self.recommended_model}")

    def _discover_finetuned_models(self) -> Dict:
        """自动发现微调模型"""
        finetuned_models = {}
        try:
            # 获取项目根目录（动态计算）
            import sys
            from pathlib import Path

            # 获取当前文件所在目录
            current_file = Path(__file__)
            # 计算项目根目录：当前文件 -> src/qa/ -> src/ -> 项目根目录
            project_root_path = current_file.parent.parent.parent

            # 微调模型目录
            finetuned_dir = project_root_path / "finetuned_models"

            # 如果动态计算失败，使用绝对路径
            if not finetuned_dir.exists():
                finetuned_dir = Path(r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetuned_models")

            if not finetuned_dir.exists():
                print(f"⚠️  微调目录不存在: {finetuned_dir}")
                return finetuned_models

            print(f"🔍 扫描微调模型目录: {finetuned_dir}")

            for model_dir in finetuned_dir.iterdir():
                if model_dir.is_dir():
                    # 检查是否有必要的模型文件
                    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
                    # 允许 .safetensors 或 .bin 格式
                    has_config = (model_dir / "config.json").exists()
                    has_model = ((model_dir / "model.safetensors").exists() or
                                 (model_dir / "pytorch_model.bin").exists())
                    has_tokenizer = (model_dir / "tokenizer.json").exists()

                    if has_config and has_model and has_tokenizer:
                        model_key = f"finetuned-{model_dir.name}"
                        finetuned_models[model_key] = {
                            "name": str(model_dir),
                            "local_path": str(model_dir),
                            "description": f"微调模型: {model_dir.name}",
                            "max_context_length": 512,
                            "recommended_top_k": 3,
                            "confidence_threshold": 0.05
                        }
                        print(f"   ✅ 发现微调模型: {model_key}")

            return finetuned_models

        except Exception as e:
            print(f"❌ 发现微调模型时出错: {e}")
            return finetuned_models

        # # 获取项目根目录（动态计算）
        # current_file = Path(__file__)
        # project_root_path = current_file.parent.parent.parent  # src/qa/ -> src/ -> project_root
        # finetuned_dir = project_root_path / "finetuned_models"
        #
        # # 或者使用绝对路径
        # # finetuned_dir = Path(r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetuned_models")
        #
        # if not finetuned_dir.exists():
        #     print(f"⚠️  微调目录不存在: {finetuned_dir}")
        #     return finetuned_models
        #
        # print(f"🔍 扫描微调模型目录: {finetuned_dir}")
        #
        # for model_dir in finetuned_dir.iterdir():
        #     if model_dir.is_dir():
        #         # 检查是否有必要的模型文件
        #         required_files = ["config.json", "model.safetensors", "tokenizer.json"]
        #         has_required = all((model_dir / f).exists() for f in required_files)
        #
        #         if has_required:
        #             model_key = f"finetuned-{model_dir.name}"
        #             finetuned_models[model_key] = {
        #                 "name": str(model_dir),
        #                 "local_path": str(model_dir),
        #                 "description": f"微调模型: {model_dir.name}",
        #                 "max_context_length": 512,
        #                 "recommended_top_k": 3,
        #                 "confidence_threshold": 0.05
        #             }
        #             print(f"   ✅ 发现微调模型: {model_key}")
        #
        # return finetuned_models

    def get_available_models(self) -> List[str]:
        """获取可用模型列表（包括微调模型）"""
        all_models = list(self.QA_MODEL_CONFIGS.keys()) + list(self.finetuned_models.keys())

        # 将微调模型排在前面
        finetuned_keys = list(self.finetuned_models.keys())
        other_keys = [m for m in all_models if m not in finetuned_keys]

        return finetuned_keys + other_keys

    def get_all_model_configs(self) -> Dict:
        """获取所有模型配置（包括微调模型）"""
        all_configs = self.QA_MODEL_CONFIGS.copy()
        all_configs.update(self.finetuned_models)
        return all_configs

    def get_model_info(self, model_key: str) -> str:
        """获取模型详细信息"""
        # 先检查预定义模型
        if model_key in self.QA_MODEL_CONFIGS:
            config = self.QA_MODEL_CONFIGS[model_key]
        elif model_key in self.finetuned_models:
            config = self.finetuned_models[model_key]
        else:
            return f"❌ 未知模型: {model_key}"

        info = f"""
    📊 模型信息: {model_key}
    • 描述: {config['description']}
    • 最大上下文: {config['max_context_length']} tokens
    • 推荐 top_k: {config.get('recommended_top_k', 3)}
    • 置信度阈值: {config.get('confidence_threshold', 0.1)}
    • 本地路径: {config['local_path']}
            """

        # 检查模型是否已加载
        if model_key in self.models:
            model_info = self.models[model_key].get_model_info()
            info += f"\n• 状态: ✅ 已加载"
            info += f"\n• 设备: {model_info['device']}"
            info += f"\n• 最大长度: {model_info['max_length']}"
            info += f"\n• 参数量: {model_info['model_parameters']:,}"
        else:
            info += f"\n• 状态: ⏳ 未加载"

            # 检查本地文件是否存在
            local_path = Path(config["local_path"])
            if local_path.exists():
                info += f"\n• 本地文件: ✅ 存在"
            else:
                info += f"\n• 本地文件: ❌ 不存在"

        return info.strip()

    def load_model(self, model_key: str, force_reload: bool = False) -> str:
        """
        加载指定模型

        Args:
            model_key: 模型键名
            force_reload: 是否强制重新加载

        Returns:
            加载状态消息
        """
        # 首先检查是否是微调模型
        if model_key in self.finetuned_models:
            config = self.finetuned_models[model_key]
            print(f"🔍 加载微调模型: {model_key}")
        elif model_key in self.QA_MODEL_CONFIGS:
            config = self.QA_MODEL_CONFIGS[model_key]
            print(f"🔍 加载预定义模型: {model_key}")
        else:
            return f"❌ 无效的模型选择: {model_key}"

        # 检查是否已加载
        if model_key in self.models and not force_reload:
            if self.current_model_key == model_key:
                return f"✅ {model_key} 已加载，无需重新加载"

        print("=" * 50)
        print(f"🔄 加载QA模型: {model_key}")
        print(f"📂 模型路径: {config['local_path']}")
        print("=" * 50)

        try:
            # 检查本地路径是否存在
            local_path = Path(config["local_path"])
            model_path = None

            if local_path.exists() and any(local_path.iterdir()):
                # 检查是否有必要的模型文件
                required_files = ["config.json", "tokenizer.json"]
                # 检查模型文件（支持两种格式）
                model_files = ["model.safetensors", "pytorch_model.bin"]

                has_required = all((local_path / f).exists() for f in required_files)
                has_model = any((local_path / f).exists() for f in model_files)

                if has_required and has_model:
                    model_path = str(local_path)
                    print(f"📁 使用本地模型: {model_path}")
                else:
                    print(f"⚠️ 本地模型文件不完整")
                    missing_files = []
                    for f in required_files:
                        if not (local_path / f).exists():
                            missing_files.append(f)
                    for f in model_files:
                        if not (local_path / f).exists():
                            missing_files.append(f)
                    print(f"   缺少文件: {missing_files}")

                    # 如果是预定义模型，尝试从网络下载
                    if model_key in self.QA_MODEL_CONFIGS:
                        model_path = config["name"]
                        print(f"🌐 尝试下载模型: {model_path}")
                    else:
                        return f"❌ 微调模型文件不完整，请重新微调"
            else:
                # 使用HuggingFace模型（仅预定义模型）
                if model_key in self.QA_MODEL_CONFIGS:
                    model_path = config["name"]
                    print(f"🌐 下载模型: {model_path}")
                else:
                    return f"❌ 微调模型路径不存在: {local_path}"

            # 加载模型
            start_time = time.time()

            try:
                model = DocumentQA(model_path)
            except Exception as e:
                print(f"❌ DocumentQA加载失败: {e}")
                # 尝试直接使用transformers加载
                print("🔄 尝试直接加载...")
                from transformers import AutoTokenizer, AutoModelForQuestionAnswering
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

                # 创建自定义的DocumentQA对象
                model = DocumentQA.__new__(DocumentQA)
                model.model_name = model_path
                model.device = "cuda" if torch.cuda.is_available() else "cpu"
                model.preprocessor = TextPreprocessor()
                model.tokenizer = tokenizer
                model.model = qa_model.to(model.device)
                model.max_length = tokenizer.model_max_length

                # 创建pipeline
                from transformers import pipeline
                model.qa_pipeline = pipeline(
                    "question-answering",
                    model=qa_model,
                    tokenizer=tokenizer,
                    device=0 if model.device == "cuda" else -1,
                    batch_size=1
                )

            load_time = time.time() - start_time

            # 更新状态
            self.models[model_key] = model
            self.current_model = model
            self.current_model_key = model_key

            # 更新默认参数
            if "confidence_threshold" in config:
                self.default_params["confidence_threshold"] = config.get("confidence_threshold", 0.1)
            if "recommended_top_k" in config:
                self.default_params["top_k"] = config.get("recommended_top_k", 3)

            print(f"✅ 模型加载成功! 耗时: {load_time:.2f}秒")
            print(f"⚙️ 推荐参数: top_k={self.default_params['top_k']}, "
                  f"置信度阈值={self.default_params['confidence_threshold']}")

            return (f"✅ {model_key} 加载成功!\n"
                    f"• 设备: {model.device}\n"
                    f"• 耗时: {load_time:.2f}秒\n"
                    f"• 推荐 top_k: {self.default_params['top_k']}")

        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            print(error_msg)

            # 提供具体建议
            if "ConnectionError" in str(e):
                error_msg += "\n💡 网络连接失败，请检查网络或使用本地模型"
            elif "401" in str(e):
                error_msg += "\n💡 认证失败，可能需要访问令牌"
            elif "404" in str(e):
                error_msg += "\n💡 模型不存在，请检查模型名称"
            elif "safetensors" in str(e):
                error_msg += "\n💡 可能是模型格式问题，尝试重新微调"

            return error_msg
    def extract_answer(
        self,
        context: str,
        question: str,
        **kwargs
    ) -> Dict:
        """
        提取答案（增强版）

        Returns:
            Dict: 包含状态、答案和统计信息
        """
        if self.current_model is None:
            return {
                "status": "error",
                "message": "请先加载模型",
                "answers": [],
                "stats": {},
                "suggestion": "请从左侧选择并加载一个模型"
            }

        try:
            start_time = time.time()

            # 合并参数
            params = self.default_params.copy()
            params.update(kwargs)

            # 验证输入
            if not context or not context.strip():
                return {
                    "status": "error",
                    "message": "文档内容不能为空",
                    "answers": [],
                    "stats": {}
                }

            if not question or not question.strip():
                return {
                    "status": "error",
                    "message": "问题不能为空",
                    "answers": [],
                    "stats": {}
                }

            print(f"🔍 开始提取答案...")
            print(f"   文档长度: {len(context)} 字符")
            print(f"   问题: {question}")
            print(f"   参数: top_k={params.get('top_k')}, "
                  f"置信度阈值={params.get('confidence_threshold')}")

            # 提取答案
            answers = self.current_model.extract_answer(
                context=context,
                question=question,
                max_answer_length=params.get("max_answer_length", 150),
                top_k=params.get("top_k", 3),
                confidence_threshold=params.get("confidence_threshold", 0.1),
                handle_long_document=params.get("handle_long_document", True)
            )

            # 计算统计信息
            process_time = time.time() - start_time
            context_length = len(context)
            question_length = len(question)

            stats = {
                "process_time": round(process_time, 2),
                "context_length": context_length,
                "question_length": question_length,
                "answers_found": len(answers),
                "model_used": self.current_model_key,
                "avg_confidence": 0
            }

            # 计算平均置信度
            if answers:
                total_confidence = sum(ans.get("confidence", 0) for ans in answers)
                stats["avg_confidence"] = round(total_confidence / len(answers), 2)

            # 生成建议
            suggestion = ""
            if not answers:
                suggestion = (
                    "💡 建议:\n"
                    "1. 确保文档包含相关信息\n"
                    "2. 尝试使用更具体的问题\n"
                    "3. 调整置信度阈值或top_k参数\n"
                    "4. 尝试其他模型（如macbert-cmrc）"
                )
            elif stats["avg_confidence"] < 20:
                suggestion = (
                    "💡 置信度较低，建议:\n"
                    "1. 检查文档与问题的相关性\n"
                    "2. 降低置信度阈值\n"
                    "3. 使用专门微调的问答模型"
                )

            if answers:
                return {
                    "status": "success",
                    "message": f"找到 {len(answers)} 个相关答案",
                    "answers": answers,
                    "stats": stats,
                    "suggestion": suggestion
                }
            else:
                return {
                    "status": "info",
                    "message": "未找到相关答案",
                    "answers": [],
                    "stats": stats,
                    "suggestion": suggestion
                }

        except Exception as e:
            print(f"❌ 提取过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

            return {
                "status": "error",
                "message": f"提取失败: {str(e)}",
                "answers": [],
                "stats": {},
                "suggestion": "请检查输入格式或尝试重新加载模型"
            }

    def batch_extract(self, contexts: List[str], questions: List[str], **kwargs) -> List[Dict]:
        """批量提取答案"""
        if self.current_model is None:
            return [{"error": "模型未加载"}]

        results = []
        for i, (context, question) in enumerate(zip(contexts, questions)):
            print(f"🔍 批量处理 {i+1}/{len(contexts)}...")
            result = self.extract_answer(context, question, **kwargs)
            results.append(result)

        return results

    def get_current_model(self) -> str:
        """获取当前模型"""
        if self.current_model_key:
            return f"{self.current_model_key} (已加载)"
        return "未加载模型"

    def get_default_params(self) -> Dict:
        """获取默认参数"""
        return self.default_params.copy()

    def clear_model(self, model_key: str = None):
        """清理模型"""
        if model_key and model_key in self.models:
            del self.models[model_key]
            if self.current_model_key == model_key:
                self.current_model = None
                self.current_model_key = None
            print(f"✅ 已清理模型: {model_key}")
        elif not model_key and self.current_model_key:
            key = self.current_model_key
            if key in self.models:
                del self.models[key]
            self.current_model = None
            self.current_model_key = None
            print(f"✅ 已清理当前模型")

    def get_recommended_model(self) -> str:
        """获取推荐模型"""
        recommended_models = ["uer-roberta-qa"]
        for model in recommended_models:
            if model in self.QA_MODEL_CONFIGS:
                return model
        return list(self.QA_MODEL_CONFIGS.keys())[0] if self.QA_MODEL_CONFIGS else "macbert-base"


# 创建全局实例
qa_manager = QAExtractorManager()