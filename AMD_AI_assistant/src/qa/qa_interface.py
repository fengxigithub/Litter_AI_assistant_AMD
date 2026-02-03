"""
文档问答界面 - 优化版Gradio Web界面
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

print("=" * 60)
print("📄 文档问答界面 - 优化版初始化")
print("=" * 60)

import gradio as gr
import json
import time
from datetime import datetime
from src.qa.qa_extractor import QAExtractorManager


class QAChat:
    """文档问答聊天类（优化版）"""

    def __init__(self):
        self.qa_manager = QAExtractorManager()
        self.conversation_history = []  # 保存问答历史
        self.max_history_size = 20  # 最大历史记录数

        # 加载推荐模型
        self.recommended_model = self.qa_manager.get_recommended_model()
        print(f"🌟 推荐模型: {self.recommended_model}")

    def get_model_list(self):
        """获取可用模型列表"""
        models = self.qa_manager.get_available_models()
        # 将推荐模型放在第一位
        if self.recommended_model in models:
            models.remove(self.recommended_model)
            models.insert(0, self.recommended_model)
        return models

    def get_model_info(self, model_key):
        """获取模型信息"""
        return self.qa_manager.get_model_info(model_key)

    def load_model(self, model_key, show_info=True):
        """加载模型"""
        result = self.qa_manager.load_model(model_key)

        if show_info and "✅" in result:
            # 获取模型详情
            model_info = self.qa_manager.get_model_info(model_key)
            result += f"\n\n{model_info}"

        return result

    def extract_answer(self, document, question, top_k=3, max_length=150,
                       confidence_threshold=0.1, handle_long_document=True):
        """提取答案（增强版）"""
        if not document.strip():
            return {
                "status": "error",
                "message": "请输入文档内容",
                "answers": [],
                "stats": {},
                "suggestion": "请在左侧输入文档内容"
            }

        if not question.strip():
            return {
                "status": "error",
                "message": "请输入问题",
                "answers": [],
                "stats": {},
                "suggestion": "请在下方输入您的问题"
            }

        # 记录开始时间
        start_time = time.time()

        # 提取答案
        result = self.qa_manager.extract_answer(
            context=document,
            question=question,
            top_k=top_k,
            max_answer_length=max_length,
            confidence_threshold=confidence_threshold,
            handle_long_document=handle_long_document
        )

        # 添加处理时间
        if "stats" in result:
            result["stats"]["total_time"] = round(time.time() - start_time, 2)

        # 保存到历史（限制历史大小）
        if result["status"] in ["success", "info"]:
            history_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "document_preview": document[:200] + ("..." if len(document) > 200 else ""),
                "question": question,
                "answers": result["answers"],
                "stats": result["stats"],
                "status": result["status"]
            }

            self.conversation_history.append(history_entry)

            # 限制历史大小
            if len(self.conversation_history) > self.max_history_size:
                self.conversation_history = self.conversation_history[-self.max_history_size:]

        return result

    def get_history_summary(self, detailed=False):
        """获取历史摘要"""
        if not self.conversation_history:
            return "📚 暂无问答历史\n\n💡 开始您的第一次问答吧！"

        summary = f"📚 最近问答历史 ({len(self.conversation_history)} 条):\n\n"

        for i, conv in enumerate(reversed(self.conversation_history[-10:]), 1):
            summary += f"🔹 {conv['timestamp']}\n"
            summary += f"   问题: {conv['question']}\n"

            if conv['answers']:
                best_answer = conv['answers'][0]
                summary += f"   最佳答案: {best_answer['text'][:80]}...\n"
                summary += f"   置信度: {best_answer.get('confidence', 0):.1f}%\n"
            else:
                summary += f"   结果: {conv['status']}\n"

            if conv.get('stats'):
                summary += f"   耗时: {conv['stats'].get('process_time', 0)}秒\n"

            summary += "-" * 50 + "\n"

        return summary.strip()

    def get_detailed_history(self):
        """获取详细历史"""
        if not self.conversation_history:
            return "暂无详细历史记录"

        detailed = "📋 详细问答历史:\n\n"
        for i, conv in enumerate(reversed(self.conversation_history), 1):
            detailed += f"记录 #{i}\n"
            detailed += f"时间: {conv['timestamp']}\n"
            detailed += f"状态: {conv['status']}\n"
            detailed += f"问题: {conv['question']}\n"

            if conv['answers']:
                detailed += "答案:\n"
                for j, ans in enumerate(conv['answers'], 1):
                    detailed += f"  {j}. {ans['text']}\n"
                    detailed += f"     置信度: {ans.get('confidence', 0):.1f}%\n"
            else:
                detailed += "答案: 未找到\n"

            detailed += "\n" + "="*60 + "\n\n"

        return detailed.strip()

    def clear_history(self):
        """清空历史"""
        self.conversation_history.clear()
        return "🗑️ 问答历史已清空"

    def get_current_model(self):
        """获取当前模型"""
        return self.qa_manager.get_current_model()

    def get_default_params(self):
        """获取默认参数"""
        return self.qa_manager.get_default_params()

    def export_history(self, format_type="json"):
        """导出历史记录"""
        if not self.conversation_history:
            return "无历史记录可导出"

        if format_type == "json":
            return json.dumps(self.conversation_history, ensure_ascii=False, indent=2)
        elif format_type == "txt":
            txt_output = "文档问答历史记录\n"
            txt_output += "=" * 40 + "\n\n"
            for conv in self.conversation_history:
                txt_output += f"时间: {conv['timestamp']}\n"
                txt_output += f"问题: {conv['question']}\n"
                txt_output += f"文档预览: {conv['document_preview']}\n"
                txt_output += f"状态: {conv['status']}\n"
                if conv['answers']:
                    txt_output += "答案:\n"
                    for ans in conv['answers']:
                        txt_output += f"  - {ans['text']} (置信度: {ans.get('confidence', 0):.1f}%)\n"
                txt_output += "\n" + "-"*40 + "\n\n"
            return txt_output
        else:
            return "不支持该格式"


def format_answers(result):
    """格式化答案输出（增强版）"""
    if "error" in result:
        return f"❌ {result['error']}"

    status = result.get("status", "")
    message = result.get("message", "")
    answers = result.get("answers", [])
    stats = result.get("stats", {})
    suggestion = result.get("suggestion", "")

    if status == "error":
        return f"❌ {message}\n\n💡 建议: {suggestion if suggestion else '请检查输入或重新加载模型'}"
    elif status == "info":
        return f"ℹ️ {message}\n\n💡 建议: {suggestion if suggestion else '请调整问题或参数后重试'}"

    # 格式化答案
    output = f"✅ {message}\n\n"

    if not answers:
        output += "⚠️ 未提取到有效答案\n"
    else:
        for i, answer in enumerate(answers, 1):
            confidence = answer.get("confidence", answer.get("score", 0) * 100)
            output += f"📄 答案 {i} (置信度: {confidence:.1f}%):\n"
            output += f"   {answer['text']}\n"
            if i < len(answers):
                output += "-" * 50 + "\n"

    # 添加统计信息
    if stats:
        output += "\n📊 统计信息:\n"
        output += f"• 处理时间: {stats.get('process_time', 0)}秒\n"
        if 'total_time' in stats:
            output += f"• 总耗时: {stats.get('total_time', 0)}秒\n"
        output += f"• 文档长度: {stats.get('context_length', 0)}字符\n"
        output += f"• 问题长度: {stats.get('question_length', 0)}字符\n"
        output += f"• 找到答案: {stats.get('answers_found', 0)}个\n"
        output += f"• 平均置信度: {stats.get('avg_confidence', 0)}%\n"
        output += f"• 使用模型: {stats.get('model_used', 'N/A')}\n"

    # 添加建议
    if suggestion:
        output += f"\n💡 优化建议:\n{suggestion}\n"

    return output


def format_stats(result):
    """格式化统计信息"""
    if "error" in result or "stats" not in result or not result["stats"]:
        return "📊 等待提取..."

    stats = result["stats"]
    stats_text = f"""
📊 处理统计:
• 耗时: {stats.get('process_time', 0)}秒
• 文档: {stats.get('context_length', 0)}字符
• 问题: {stats.get('question_length', 0)}字符
• 答案: {stats.get('answers_found', 0)}个
• 置信度: {stats.get('avg_confidence', 0)}%
• 模型: {stats.get('model_used', 'N/A')}
"""
    return stats_text.strip()


def create_qa_interface():
    """创建QA界面（增强版）"""
    qa_chat = QAChat()
    default_params = qa_chat.get_default_params()

    with gr.Blocks(title="📄 文档问答助手 - 完整版") as demo:
        gr.Markdown("""
        # 📄 文档问答助手 - 完整版
        ### 支持文档问答 + 模型微调功能
        """)

        with gr.Tabs():
            # ===== 第一个标签页：文档问答 =====
            with gr.Tab("💬 文档问答"):
                with gr.Row():
                    # 左侧控制面板
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 模型控制")

                        # 模型选择
                        model_selector = gr.Dropdown(
                            choices=qa_chat.get_model_list(),
                            value=qa_chat.get_model_list()[0] if qa_chat.get_model_list() else None,
                            label="选择QA模型",
                            interactive=True,
                            info="🌟 推荐使用已微调的模型"
                        )

                        model_info = gr.Textbox(
                            label="📊 模型信息",
                            value=qa_chat.get_model_info(
                                qa_chat.get_model_list()[0]) if qa_chat.get_model_list() else "无可用模型",
                            lines=8,
                            interactive=False
                        )

                        with gr.Row():
                            load_btn = gr.Button("🚀 加载模型", variant="primary", scale=2)
                            refresh_models_btn = gr.Button("🔄 刷新", scale=1)

                        # 参数设置
                        gr.Markdown("### ⚙️ 参数设置")

                        with gr.Accordion("高级参数设置", open=False):
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=default_params.get("top_k", 3),
                                step=1,
                                label="返回答案数量 (top_k)"
                            )

                            max_length_slider = gr.Slider(
                                minimum=50,
                                maximum=300,
                                value=default_params.get("max_answer_length", 150),
                                step=10,
                                label="最大答案长度"
                            )

                            confidence_slider = gr.Slider(
                                minimum=0.01,
                                maximum=0.5,
                                value=default_params.get("confidence_threshold", 0.1),
                                step=0.01,
                                label="置信度阈值",
                                info="值越低，返回的答案越多"
                            )

                            handle_long_doc = gr.Checkbox(
                                label="处理长文档",
                                value=default_params.get("handle_long_document", True),
                                info="自动分割长文档"
                            )

                        status_display = gr.Textbox(
                            label="📈 模型状态",
                            value="请选择模型并点击加载",
                            lines=3,
                            interactive=False
                        )

                        gr.Markdown("---")

                        # 历史管理
                        gr.Markdown("### 📚 问答历史")

                        history_display = gr.Textbox(
                            label="历史记录",
                            value=qa_chat.get_history_summary(),
                            lines=8,
                            interactive=False
                        )

                        with gr.Row():
                            refresh_history_btn = gr.Button("🔄 刷新历史")
                            clear_history_btn = gr.Button("🗑️ 清空历史", variant="stop")
                            export_btn = gr.Button("📥 导出历史")

                        current_model_display = gr.Textbox(
                            label="🤖 当前模型",
                            value="未加载",
                            interactive=False
                        )

                    # 右侧问答区域
                    with gr.Column(scale=2):
                        gr.Markdown("### 📄 文档问答区域")

                        # 示例按钮
                        with gr.Row():
                            example_btn1 = gr.Button("📋 示例1: 公司介绍", size="sm")
                            example_btn2 = gr.Button("📋 示例2: 技术文档", size="sm")
                            example_btn3 = gr.Button("📋 示例3: 新闻内容", size="sm")

                        # 文档输入
                        document_input = gr.Textbox(
                            label="📄 文档内容",
                            placeholder="请在此处粘贴或输入文档内容...\n（支持长文档，系统会自动分割处理）",
                            lines=15,
                            max_lines=30,
                            info="建议文档长度在100-5000字符之间"
                        )

                        # 问题输入
                        question_input = gr.Textbox(
                            label="❓ 问题",
                            placeholder="请输入您的问题...\n例如：这篇文章的主要内容是什么？",
                            lines=3
                        )

                        with gr.Row():
                            extract_btn = gr.Button("🔍 提取答案", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清空输入", scale=1)

                        # 答案输出
                        answer_output = gr.Textbox(
                            label="📝 提取结果",
                            placeholder="答案将显示在这里...",
                            lines=20,
                            interactive=False
                        )

                        # 统计信息
                        stats_display = gr.Textbox(
                            label="📊 处理统计",
                            value="等待提取...",
                            lines=4,
                            interactive=False
                        )

            # ===== 第二个标签页：模型微调 =====
            with gr.Tab("🔧 模型微调"):
                gr.Markdown("""
                ### 🎯 模型微调设置
                **功能说明**：在现有模型基础上进行微调，使其更适合您的文档类型
                **注意**：微调需要较长时间，建议准备好数据集后再开始
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 基础设置")

                        base_model_select = gr.Dropdown(
                            choices=["macbert-base", "bert-wwm-base"],
                            value="macbert-base",
                            label="选择基础模型",
                            info="选择要微调的基础模型"
                        )

                        dataset_path = gr.Textbox(
                            label="数据集路径",
                            value=r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetune_data\custom_sample",
                            placeholder="输入数据集路径",
                            info="包含train.json, dev.json, test.json的目录"
                        )

                        output_dir = gr.Textbox(
                            label="输出目录",
                            value=r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetuned_models",
                            placeholder="微调模型输出目录",
                            info="微调后的模型将保存到此目录"
                        )

                        model_name_suffix = gr.Textbox(
                            label="模型名称后缀",
                            value="my_finetuned",
                            placeholder="如: my_finetuned",
                            info="将添加到模型名称后面，用于区分"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 训练参数")

                        epochs = gr.Slider(
                            minimum=1, maximum=10, value=3,
                            step=1, label="训练轮数 (epochs)",
                            info="建议3-5轮"
                        )

                        batch_size = gr.Slider(
                            minimum=1, maximum=16, value=8,
                            step=1, label="批次大小 (batch_size)",
                            info="根据显存调整"
                        )

                        learning_rate = gr.Number(
                            value=3e-5, label="学习率 (learning_rate)",
                            info="建议3e-5"
                        )

                        device_select = gr.Dropdown(
                            choices=["auto", "cpu", "directml", "cuda"],
                            value="auto",
                            label="训练设备",
                            info="auto: 自动检测, directml: AMD显卡"
                        )

                gr.Markdown("---")

                with gr.Row():
                    start_finetune_btn = gr.Button("🚀 生成微调命令", variant="primary", size="lg")
                    show_finetune_cmd_btn = gr.Button("📋 显示微调命令", size="lg")

                finetune_output = gr.Textbox(
                    label="微调输出/命令",
                    lines=15,
                    interactive=False,
                    placeholder="微调命令将显示在这里..."
                )

                # 微调进度（模拟）
                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="训练进度", interactive=False,
                    visible=False  # 暂时隐藏
                )

                gr.Markdown("---")

                with gr.Row():
                    refresh_finetuned_btn = gr.Button("🔄 刷新微调模型列表")
                    open_finetune_dir_btn = gr.Button("📁 打开微调目录")

                finetuned_models_list = gr.Textbox(
                    label="已发现的微调模型",
                    lines=8,
                    interactive=False,
                    value="点击'刷新微调模型列表'查看"
                )

                gr.Markdown("""
                ### 💡 微调使用说明
                1. **准备数据集**：将您的文档和问答对整理成CMRC2018格式
                2. **设置参数**：调整训练参数，选择合适的设备
                3. **生成命令**：点击"生成微调命令"获取训练命令
                4. **运行微调**：在终端/命令行中运行生成的命令
                5. **加载模型**：微调完成后，刷新模型列表并加载新模型

                **数据集格式**：需要三个文件：`train.json`、`dev.json`、`test.json`
                """)

        # ===== 事件绑定 =====

        # 页面加载时初始化
        def on_page_load():
            return qa_chat.get_history_summary()

        demo.load(
            on_page_load,
            outputs=[history_display]
        )

        # 1. 模型选择器更新信息
        def update_model_info(model_key):
            return qa_chat.get_model_info(model_key)

        model_selector.change(
            update_model_info,
            inputs=[model_selector],
            outputs=[model_info]
        )

        # 2. 加载模型
        def on_load_model(model_key):
            status = qa_chat.load_model(model_key)
            current_model = qa_chat.get_current_model()
            return status, current_model, qa_chat.get_history_summary()

        load_btn.click(
            on_load_model,
            inputs=[model_selector],
            outputs=[status_display, current_model_display, history_display]
        )

        # 3. 刷新模型列表
        def refresh_model_list():
            return gr.update(choices=qa_chat.get_model_list()), \
                qa_chat.get_model_info(qa_chat.get_model_list()[0]) if qa_chat.get_model_list() else "无可用模型"

        refresh_models_btn.click(
            refresh_model_list,
            outputs=[model_selector, model_info]
        )

        # 4. 提取答案
        def on_extract_answer(document, question, top_k, max_length, confidence, handle_long):
            if not document.strip() or not question.strip():
                return "请输入文档内容和问题", "等待提取...", qa_chat.get_current_model()

            # 提取答案
            result = qa_chat.extract_answer(
                document, question, top_k, max_length, confidence, handle_long
            )

            # 格式化输出
            formatted_answer = format_answers(result)
            stats_text = format_stats(result)

            return formatted_answer, stats_text, qa_chat.get_current_model()

        extract_btn.click(
            on_extract_answer,
            inputs=[document_input, question_input, top_k_slider, max_length_slider,
                    confidence_slider, handle_long_doc],
            outputs=[answer_output, stats_display, current_model_display]
        )

        # 5. 清空输入
        clear_btn.click(
            lambda: ("", "", "", "等待提取...", qa_chat.get_current_model()),
            outputs=[document_input, question_input, answer_output,
                     stats_display, current_model_display]
        )

        # 6. 示例按钮
        def load_example(example_id):
            examples = {
                1: {
                    "document": """阿里巴巴集团成立于1999年，是一家以电子商务为核心业务的互联网公司。公司总部位于中国杭州，业务涵盖电商、云计算、数字媒体和娱乐等多个领域。

阿里巴巴的使命是让天下没有难做的生意。通过淘宝、天猫等平台，公司为数百万商家和数亿消费者提供交易服务。此外，阿里云已成为全球领先的云计算服务提供商之一。

2023财年，阿里巴巴集团总营收达到8686亿元，净利润为725亿元。公司持续投资于技术创新，特别是在人工智能和大数据领域。""",
                    "question": "阿里巴巴的总部在哪里？"
                },
                2: {
                    "document": """深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。神经网络由输入层、隐藏层和输出层组成，每层包含多个神经元。

反向传播算法是训练神经网络的关键技术，通过计算损失函数的梯度来更新网络权重。常用的激活函数包括Sigmoid、ReLU和Tanh。

卷积神经网络（CNN）专门用于处理图像数据，而循环神经网络（RNN）则擅长处理序列数据。Transformer架构近年来在自然语言处理领域取得了显著成功。""",
                    "question": "什么是卷积神经网络的主要应用？"
                },
                3: {
                    "document": """中国国家航天局近日宣布，嫦娥六号探测器成功在月球背面着陆。这是人类历史上首次在月球背面进行的采样返回任务。

嫦娥六号任务的主要科学目标包括：采集月球背面的土壤和岩石样本，进行现场分析，并将样本返回地球。探测器携带了多种科学仪器，包括全景相机、光谱仪和探地雷达。

此次任务的成功实施，标志着中国在深空探测领域取得了重要进展，为未来的月球科研站建设和载人登月任务奠定了坚实基础。""",
                    "question": "嫦娥六号的主要任务是什么？"
                }
            }

            if example_id in examples:
                example = examples[example_id]
                return example["document"], example["question"]
            return "", ""

        example_btn1.click(lambda: load_example(1), outputs=[document_input, question_input])
        example_btn2.click(lambda: load_example(2), outputs=[document_input, question_input])
        example_btn3.click(lambda: load_example(3), outputs=[document_input, question_input])

        # 7. 历史管理
        def refresh_history():
            return qa_chat.get_history_summary()

        refresh_history_btn.click(
            refresh_history,
            outputs=[history_display]
        )

        clear_history_btn.click(
            lambda: (qa_chat.clear_history(), qa_chat.get_history_summary()),
            outputs=[status_display, history_display]
        )

        def on_export_history():
            return qa_chat.export_history("txt")

        export_btn.click(
            on_export_history,
            outputs=[answer_output]
        )

        # ===== 微调相关事件 =====

        # 8. 生成微调命令
        def generate_finetune_command(base_model, dataset, output, suffix, epochs_val, batch_size_val, lr, device):
            """生成微调命令"""
            import datetime

            # 构建输出路径
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_output = f"{output}/{base_model.replace('-base', '')}_{suffix}_{timestamp}"

            # 构建命令
            cmd = f"""# 微调命令 - 复制到终端运行
cd /d "F:\\py_work\\AMD_AI_Project\\AMD_AI_Project"

python finetune_scripts/finetune_qa.py \\
  --base_model models/qa_models/{base_model.split('-')[0]} \\
  --dataset "{dataset}" \\
  --output_dir "{model_output}" \\
  --epochs {epochs_val} \\
  --batch_size {batch_size_val} \\
  --learning_rate {lr} \\
  --device {device}

# 命令说明：
# 1. 请确保已激活虚拟环境: .venv\\Scripts\\activate
# 2. 确保数据集路径正确
# 3. 微调完成后，刷新模型列表即可看到新模型
# 4. 模型将保存到: {model_output}
"""
            return cmd

        start_finetune_btn.click(
            generate_finetune_command,
            inputs=[base_model_select, dataset_path, output_dir, model_name_suffix,
                    epochs, batch_size, learning_rate, device_select],
            outputs=[finetune_output]
        )

        # 9. 显示简化的微调命令
        def show_finetune_command():
            cmd = """# 基本微调命令
python finetune_scripts/finetune_qa.py \\
  --base_model models/qa_models/macbert \\
  --dataset finetune_data/custom_sample \\
  --output_dir finetuned_models/macbert_finetuned \\
  --epochs 3 \\
  --batch_size 8 \\
  --learning_rate 3e-5 \\
  --device auto

# 准备数据集示例：
# python src/qa/finetune_data.py
"""
            return cmd

        show_finetune_cmd_btn.click(
            show_finetune_command,
            outputs=[finetune_output]
        )

        # 10. 刷新微调模型列表
        def refresh_finetuned_list():
            import glob
            import os

            finetuned_dir = r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetuned_models"

            if not os.path.exists(finetuned_dir):
                return "微调目录不存在，请先创建: finetuned_models/"

            models = glob.glob(f"{finetuned_dir}/*")

            if not models:
                return "未找到微调模型\n\n请先运行微调训练生成模型"

            model_list = "📁 已发现的微调模型:\n\n"
            for model_path in models:
                if os.path.isdir(model_path):
                    model_name = os.path.basename(model_path)

                    # 检查是否是有效模型
                    required_files = ["config.json", "pytorch_model.bin"]
                    is_valid = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)

                    status = "✅ 有效" if is_valid else "⚠️ 不完整"
                    model_list += f"• {model_name} ({status})\n"

            model_list += f"\n💡 共发现 {len([m for m in models if os.path.isdir(m)])} 个模型"
            model_list += "\n💡 刷新问答页面的模型列表即可看到这些模型"

            return model_list

        refresh_finetuned_btn.click(
            refresh_finetuned_list,
            outputs=[finetuned_models_list]
        )

        # 11. 打开微调目录
        def open_finetune_directory():
            import os
            import subprocess

            finetuned_dir = r"F:\py_work\AMD_AI_Project\AMD_AI_Project\finetuned_models"

            if not os.path.exists(finetuned_dir):
                os.makedirs(finetuned_dir, exist_ok=True)

            try:
                subprocess.Popen(f'explorer "{finetuned_dir}"')
                return f"✅ 已打开微调目录:\n{finetuned_dir}"
            except Exception as e:
                return f"❌ 打开目录失败: {str(e)}"

        open_finetune_dir_btn.click(
            open_finetune_directory,
            outputs=[finetune_output]
        )

    return demo


def main():
    """主函数"""
    print("🌐 启动文档问答助手 - 优化版...")
    print("💡 新增功能:")
    print("  • 文本预处理和清洗")
    print("  • 长文档智能分块")
    print("  • 答案置信度阈值")
    print("  • 答案后处理")
    print("  • 示例文档和问题")
    print("  • 历史记录导出")
    print("-" * 50)
    print(f"💡 本地访问: http://127.0.0.1:7861")
    print(f"🌐 局域网访问: http://192.168.1.4:7861")
    print("-" * 50)
    print("⚠️  如果模型加载失败，请确保:")
    print("   1. 网络连接正常")
    print("   2. 模型路径正确")
    print("   3. 有足够的磁盘空间")

    demo = create_qa_interface()

    demo.launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        debug=False,
        favicon_path=None
    )


if __name__ == "__main__":
    main()