"""
微调数据准备脚本
支持CMRC2018格式和自定义格式
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random


class QADataProcessor:
    """问答数据处理类"""

    @staticmethod
    def load_cmrc2018_data(data_dir: str) -> Dict:
        """
        加载CMRC2018格式数据

        CMRC2018格式:
        {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "文章内容...",
                            "qas": [
                                {
                                    "question": "问题...",
                                    "answers": [
                                        {"text": "答案", "answer_start": 123}
                                    ],
                                    "id": "unique_id"
                                }
                            ]
                        }
                    ],
                    "title": "标题"
                }
            ]
        }
        """
        data_path = Path(data_dir)

        train_data = {}
        dev_data = {}
        test_data = {}

        # 尝试加载不同版本的数据
        possible_files = {
            "train": ["train.json", "cmrc2018_train.json", "train-v1.1.json"],
            "dev": ["dev.json", "cmrc2018_dev.json", "dev-v1.1.json"],
            "test": ["test.json", "cmrc2018_test.json", "test-v1.1.json"]
        }

        for split, filenames in possible_files.items():
            for filename in filenames:
                file_path = data_path / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if split == "train":
                            train_data = json.load(f)
                        elif split == "dev":
                            dev_data = json.load(f)
                        elif split == "test":
                            test_data = json.load(f)
                    print(f"✅ 加载{split}数据: {file_path}")
                    break

        return {
            "train": train_data,
            "dev": dev_data,
            "test": test_data
        }

    @staticmethod
    def convert_to_squad_format(data: Dict) -> List[Dict]:
        """转换为SQuAD格式"""
        squad_examples = []

        if "data" not in data:
            return squad_examples

        for article in data["data"]:
            for paragraph in article.get("paragraphs", []):
                context = paragraph["context"]
                for qa in paragraph.get("qas", []):
                    question = qa["question"]
                    answers = qa.get("answers", [])

                    example = {
                        "context": context,
                        "question": question,
                        "id": qa.get("id", f"q_{len(squad_examples)}"),
                    }

                    if answers:
                        # 训练数据
                        example["answers"] = answers
                    else:
                        # 测试数据
                        example["answers"] = [{"text": "", "answer_start": 0}]

                    squad_examples.append(example)

        return squad_examples

    @staticmethod
    def create_custom_dataset(documents: List[str],
                              qa_pairs: List[Dict],
                              output_path: str):
        """
        创建自定义数据集

        Args:
            documents: 文档列表
            qa_pairs: [{"question": "...", "answer": "...", "context_index": 0}]
        """
        data = {"data": []}

        for doc_idx, context in enumerate(documents):
            # 找到与该文档相关的问题
            doc_questions = [qa for qa in qa_pairs
                             if qa.get("context_index") == doc_idx]

            if not doc_questions:
                continue

            qas = []
            for qa_idx, qa in enumerate(doc_questions):
                # 在上下文中查找答案位置
                answer_text = qa["answer"]
                answer_start = context.find(answer_text)

                if answer_start == -1:
                    print(f"⚠️  答案未在上下文中找到: {answer_text[:50]}...")
                    answer_start = 0

                qas.append({
                    "question": qa["question"],
                    "answers": [{
                        "text": answer_text,
                        "answer_start": answer_start
                    }],
                    "id": f"doc{doc_idx}_qa{qa_idx}"
                })

            if qas:
                data["data"].append({
                    "title": f"Document_{doc_idx}",
                    "paragraphs": [{
                        "context": context,
                        "qas": qas
                    }]
                })

        # 保存数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存自定义数据集: {output_path}")
        print(f"   文档数: {len(documents)}")
        print(f"   问题数: {len(qa_pairs)}")

        return data

    @staticmethod
    def split_dataset(data: Dict, train_ratio: float = 0.8,
                      dev_ratio: float = 0.1) -> Dict[str, Dict]:
        """分割数据集"""
        all_examples = []

        for article in data.get("data", []):
            for paragraph in article.get("paragraphs", []):
                context = paragraph["context"]
                for qa in paragraph.get("qas", []):
                    all_examples.append({
                        "article": article,
                        "paragraph": paragraph,
                        "context": context,
                        "qa": qa
                    })

        # 随机打乱
        random.shuffle(all_examples)

        train_size = int(len(all_examples) * train_ratio)
        dev_size = int(len(all_examples) * dev_ratio)

        train_examples = all_examples[:train_size]
        dev_examples = all_examples[train_size:train_size + dev_size]
        test_examples = all_examples[train_size + dev_size:]

        # 重新组装数据
        def build_data(examples):
            articles = {}
            for ex in examples:
                title = ex["article"]["title"]
                if title not in articles:
                    articles[title] = {
                        "title": title,
                        "paragraphs": []
                    }

                # 检查是否已存在相同的段落
                paragraph_exists = False
                for p in articles[title]["paragraphs"]:
                    if p["context"] == ex["context"]:
                        p["qas"].append(ex["qa"])
                        paragraph_exists = True
                        break

                if not paragraph_exists:
                    articles[title]["paragraphs"].append({
                        "context": ex["context"],
                        "qas": [ex["qa"]]
                    })

            return {"data": list(articles.values())}

        return {
            "train": build_data(train_examples),
            "dev": build_data(dev_examples),
            "test": build_data(test_examples)
        }

    @staticmethod
    def prepare_qa_pairs_for_finetuning(contexts: List[str],
                                        qa_pairs: List[Dict],
                                        output_dir: str):
        """准备微调数据"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建完整数据集
        full_data = QADataProcessor.create_custom_dataset(
            contexts, qa_pairs,
            str(output_path / "full_dataset.json")
        )

        # 分割数据集
        splits = QADataProcessor.split_dataset(full_data)

        for split_name, split_data in splits.items():
            filename = output_path / f"{split_name}_dataset.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存{split_name}数据集: {filename}")

        # 生成统计信息
        stats = {
            "total_contexts": len(contexts),
            "total_qa_pairs": len(qa_pairs),
            "train_examples": len(splits["train"].get("data", [])),
            "dev_examples": len(splits["dev"].get("data", [])),
            "test_examples": len(splits["test"].get("data", []))
        }

        stats_path = output_path / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return splits


def prepare_sample_dataset():
    """准备示例数据集（用于测试）"""
    sample_contexts = [
        "阿里巴巴集团成立于1999年，是一家以电子商务为核心业务的互联网公司。"
        "公司总部位于中国杭州，业务涵盖电商、云计算、数字媒体和娱乐等多个领域。",

        "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。"
        "神经网络由输入层、隐藏层和输出层组成，每层包含多个神经元。"
    ]

    sample_qa_pairs = [
        {
            "question": "阿里巴巴的总部在哪里？",
            "answer": "杭州",
            "context_index": 0
        },
        {
            "question": "阿里巴巴是哪一年成立的？",
            "answer": "1999年",
            "context_index": 0
        },
        {
            "question": "什么是深度学习？",
            "answer": "机器学习的一个分支",
            "context_index": 1
        },
        {
            "question": "神经网络由哪些层组成？",
            "answer": "输入层、隐藏层和输出层",
            "context_index": 1
        }
    ]

    processor = QADataProcessor()
    output_dir = r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetune_data\custom_sample"

    return processor.prepare_qa_pairs_for_finetuning(
        sample_contexts, sample_qa_pairs, output_dir
    )


# 在 finetune_data.py 中添加简单的测试数据集
# 在 finetune_data.py 中添加简单的测试数据集
def create_simple_test_dataset():
    """创建简单的测试数据集"""
    data = {
        "data": [
            {
                "title": "测试文档",
                "paragraphs": [
                    {
                        "context": "苹果公司是一家美国科技公司，总部位于加利福尼亚州的库比蒂诺。",
                        "qas": [
                            {
                                "question": "苹果公司的总部在哪里？",
                                "answers": [
                                    {
                                        "text": "加利福尼亚州的库比蒂诺",
                                        "answer_start": 14
                                    }
                                ],
                                "id": "q1"
                            }
                        ]
                    },
                    {
                        "context": "Python是一种高级编程语言，由Guido van Rossum于1991年创建。",
                        "qas": [
                            {
                                "question": "Python是谁创建的？",
                                "answers": [
                                    {
                                        "text": "Guido van Rossum",
                                        "answer_start": 16
                                    }
                                ],
                                "id": "q2"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    output_dir = Path(r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetune_data\simple_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存为训练集和测试集
    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 复制一份作为测试集
    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 创建简单测试数据集: {output_dir}")
    return str(output_dir)

if __name__ == "__main__":
    print("准备示例数据集...")
    prepare_sample_dataset()
    print("✅ 示例数据集准备完成!")