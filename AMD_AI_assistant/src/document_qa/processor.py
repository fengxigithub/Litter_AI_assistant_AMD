#!/usr/bin/env python3
"""
文档处理器 - 集成到现有项目
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict


class DocumentProcessor:
    """轻量级文档处理器"""

    def __init__(self, base_dir: str = None):
        """
        初始化

        Args:
            base_dir: 项目根目录路径，None则自动检测
        """
        if base_dir:
            self.project_root = Path(base_dir)
        else:
            # 自动检测项目根目录（虚拟环境所在目录的父目录）
            current_file = Path(__file__).resolve()
            # 向上找到项目根目录（包含src目录的目录）
            while current_file.parent.name != 'src' and current_file.parent != current_file:
                current_file = current_file.parent
            self.project_root = current_file.parent.parent

        # 设置目录
        self.documents_dir = self.project_root / "documents"
        self.training_dir = self.project_root / "data"  # 复用现有data目录
        self.models_dir = self.project_root / "models"

        # 创建目录
        self.documents_dir.mkdir(exist_ok=True)
        self.training_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        print(f"📁 项目根目录: {self.project_root}")
        print(f"📁 文档目录: {self.documents_dir}")
        print(f"📁 训练数据: {self.training_dir}")

    def extract_text_from_file(self, file_path: Path) -> str:
        """从文件中提取文本（支持txt和简单文本）"""
        try:
            # 检查文件类型
            suffix = file_path.suffix.lower()

            if suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif suffix == '.pdf':
                # 尝试使用PyPDF2
                try:
                    import PyPDF2
                    text = ""
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    print("⚠️  需要安装PyPDF2: pip install PyPDF2")
                    return ""
            elif suffix in ['.docx', '.doc']:
                # 尝试使用python-docx
                try:
                    from docx import Document
                    doc = Document(file_path)
                    return "\n".join([para.text for para in doc.paragraphs])
                except ImportError:
                    print("⚠️  需要安装python-docx: pip install python-docx")
                    return ""
            else:
                # 尝试按文本文件读取
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except:
                    print(f"⚠️  不支持的文件格式: {suffix}")
                    return ""

        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""

        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留中文和标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。？！：；"\'()（）【】《》、.,!?;:]', ' ', text)
        return text.strip()

    def split_into_sections(self, text: str, min_section_length: int = 200) -> List[str]:
        """将文本分割成有意义的段落"""
        sections = []

        # 按句子分割（中文标点）
        sentences = re.split(r'[。！？!?\.]\s*', text)

        current_section = ""
        for sentence in sentences:
            if not sentence.strip():
                continue

            current_section += sentence + "。"

            # 如果达到最小长度，开始新段落
            if len(current_section) >= min_section_length:
                sections.append(current_section.strip())
                current_section = ""

        # 添加最后一段
        if current_section:
            sections.append(current_section.strip())

        return sections

    def generate_qa_from_section(self, section: str, section_id: int) -> List[Dict]:
        """从文本段落生成问答对"""
        qa_pairs = []

        if len(section) < 50:  # 太短的段落跳过
            return qa_pairs

        # 1. 总结性问题
        summary_q = f"请总结这一段内容"
        summary_a = f"这一段的主要内容是：{section[:200]}..."
        qa_pairs.append({
            "instruction": summary_q,
            "response": summary_a,
            "type": "summary",
            "source": f"section_{section_id}"
        })

        # 2. 提取关键信息（句子级别）
        sentences = [s.strip() for s in section.split('。') if s.strip()]
        for i, sentence in enumerate(sentences[:3]):  # 取前3个句子
            if len(sentence) > 20:
                detail_q = f"关于'{sentence[:30]}...'的具体内容是什么？"
                detail_a = sentence
                qa_pairs.append({
                    "instruction": detail_q,
                    "response": detail_a,
                    "type": "detail",
                    "source": f"section_{section_id}_sentence_{i}"
                })

        # 3. 术语解释
        # 提取2-4字的中文词语作为可能术语
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', section)
        for word in list(set(chinese_words))[:3]:  # 取前3个不重复的词语
            # 找到包含这个词的上下文
            context_sentences = [s for s in sentences if word in s]
            if context_sentences:
                term_q = f"什么是'{word}'？"
                term_a = context_sentences[0]
                qa_pairs.append({
                    "instruction": term_q,
                    "response": term_a,
                    "type": "term",
                    "source": f"section_{section_id}_term_{word}"
                })

        return qa_pairs

    def process_documents(self, generate_qa: bool = True) -> str:
        """处理文档并生成训练数据"""
        print("=" * 60)
        print("📚 文档处理系统")
        print("=" * 60)

        # 检查文档目录
        if not self.documents_dir.exists():
            print(f"📁 创建文档目录: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True)

        # 查找文档文件
        supported_extensions = ['.txt', '.pdf', '.docx', '.doc']
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(list(self.documents_dir.glob(f"*{ext}")))

        if not doc_files:
            print("⚠️  未找到文档文件")
            print(f"💡 请将文档放入: {self.documents_dir}")
            print(f"📄 支持格式: {', '.join(supported_extensions)}")
            return None

        print(f"📁 找到 {len(doc_files)} 个文档:")
        for f in doc_files:
            print(f"  • {f.name}")

        all_qa_pairs = []
        all_sections = []

        # 处理每个文档
        for doc_file in doc_files:
            print(f"\n📄 处理: {doc_file.name}")

            # 提取文本
            text = self.extract_text_from_file(doc_file)
            if not text:
                print(f"  ⚠️  无法提取文本，跳过")
                continue

            # 清理文本
            cleaned_text = self.clean_text(text)
            print(f"  📝 原始字符: {len(text):,} → 清理后: {len(cleaned_text):,}")

            if not cleaned_text:
                print(f"  ⚠️  清理后无内容，跳过")
                continue

            # 分割成段落
            sections = self.split_into_sections(cleaned_text)
            print(f"  📊 分割成 {len(sections)} 个段落")

            all_sections.extend(sections)

            # 生成问答对
            if generate_qa:
                for i, section in enumerate(sections):
                    qa_pairs = self.generate_qa_from_section(section, i)
                    all_qa_pairs.extend(qa_pairs)

        # 保存结果
        if generate_qa and all_qa_pairs:
            # 保存训练数据
            training_file = self.training_dir / "document_qa_data.jsonl"
            with open(training_file, 'w', encoding='utf-8') as f:
                for qa in all_qa_pairs:
                    f.write(json.dumps(qa, ensure_ascii=False) + '\n')

            print(f"\n✅ 处理完成！")
            print(f"📊 文档段落: {len(all_sections)} 个")
            print(f"📊 生成问答: {len(all_qa_pairs)} 对")
            print(f"📁 训练数据: {training_file}")

            return str(training_file)

        elif all_sections:
            # 只保存文本
            text_file = self.training_dir / "document_texts.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for i, section in enumerate(all_sections):
                    f.write(f"=== 段落 {i + 1} ===\n")
                    f.write(section + "\n\n")

            print(f"\n✅ 文本提取完成")
            print(f"📊 文档段落: {len(all_sections)} 个")
            print(f"📁 文本文件: {text_file}")

            return str(text_file)

        return None

    def quick_start(self):
        """快速启动文档处理"""
        print("🎯 快速启动文档训练")

        # 1. 检查依赖
        try:
            import transformers
            import torch
            print("✅ 核心依赖已安装")
        except ImportError:
            print("❌ 请先安装核心依赖")
            print("💡 运行: pip install transformers torch")
            return

        # 2. 处理文档
        training_file = self.process_documents(generate_qa=True)

        if training_file:
            print("\n🚀 下一步:")
            print(f"运行文档训练: python -m src.document_qa.trainer --data {training_file}")
            print("\n💡 或使用现有训练系统:")
            print(f"python train.py  # 选择文件: {training_file}")

        return training_file