"""
AMD 7900XTX AI助手 - 完整功能版
支持模型切换 + 对话记忆
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PROJECT_ROOT, MODELS_DIR, CACHE_DIR, DATA_DIR,
    LOCAL_MODEL_PATHS,HF_MODELS, UI_CONFIG
)

print("=" * 60)
print("🚀 AMD 7900XTX AI助手 - 完整功能版")
print("=" * 60)
print(f"📁 项目根目录: {PROJECT_ROOT}")
print(f"📁 模型目录: {MODELS_DIR}")
print(f"📁 缓存目录: {CACHE_DIR}")
print(f"📁 数据目录: {DATA_DIR}")
print("=" * 60)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "huggingface")

import gradio as gr
import torch
import webbrowser
import json
import time
import pickle
from pathlib import Path
from datetime import datetime

# 检查DirectML
try:
    import torch_directml
    DML_AVAILABLE = True
    print("✅ DirectML可用")
except ImportError:
    DML_AVAILABLE = False
    print("⚠️  DirectML未安装")

print(f"🔧 PyTorch版本: {torch.__version__}")
print(f"🔧 Gradio版本: {gr.__version__}")
print()

# ——————————————————————————————————————记忆管理器
class MemoryManager:
    """对话记忆管理器"""

    def __init__(self, memory_file=UI_CONFIG["memory_file"], max_memory=UI_CONFIG["max_memory_items"]):
        self.memory_file = Path(memory_file)
        self.max_memory = max_memory
        self.conversations = self.load_memory()

        print(f"📝 记忆文件: {self.memory_file}")
        print(f"📝 最大记忆条数: {self.max_memory}")

    def load_memory(self):
        """加载对话记忆"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ 加载对话记忆: {len(data)} 条记录")
                return data
            except Exception as e:
                print(f"⚠️  记忆加载失败: {e}")
        return []

    def save_memory(self):
        """保存对话记忆"""
        try:
            if len(self.conversations) > self.max_memory:
                self.conversations = self.conversations[-self.max_memory:]

            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.conversations, f)
            return True
        except Exception as e:
            print(f"❌ 记忆保存失败: {e}")
            return False

    def add_conversation(self, user_message, ai_response, model_used):
        """添加对话记录"""
        conversation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_message,
            "ai": ai_response,
            "model": model_used,
            "tokens": len(ai_response.split())
        }

        self.conversations.append(conversation)
        self.save_memory()

        return self.format_conversation(conversation)

    def format_conversation(self, conv):
        """格式化单条对话记录"""
        return f"[{conv['timestamp']}] {conv['model']}\n👤 {conv['user'][:50]}...\n🤖 {conv['ai'][:100]}...\n"

    def get_recent_memory(self, count=5):
        """获取最近的对话记忆"""
        recent = self.conversations[-count:] if self.conversations else []
        if recent:
            return "📚 最近对话:\n" + "\n".join([self.format_conversation(c) for c in recent])
        return "📚 暂无对话历史"

    def clear_memory(self):
        """清空对话记忆"""
        self.conversations = []
        if self.memory_file.exists():
            self.memory_file.unlink()
        print("✅ 对话记忆已清空")
        return "🗑️ 对话记忆已清空"

# ————————————————————————————ModelManager类
class ModelManager:
    """模型管理器"""

    def __init__(self):
        # 使用全局配置
        self.project_root = PROJECT_ROOT
        self.models_dir = MODELS_DIR

        # 可用模型配置（动态合并）
        self.available_models = self._init_models()

        self.current_model = None
        self.current_model_key = "Qwen2.5-0.5B"  # 默认模型
        self.device = None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        print(f"📊 可用的模型:")
        for model_key, info in self.available_models.items():
            status = "✅" if info.get("type") != "local" or self._check_model_exists(info["name"]) else "⚠️ "
            print(f"  {status} {model_key} ({info['type']})")

    def _init_models(self):
        """初始化模型配置（合并HF和本地模型）"""
        models = {}



        # 2. 添加本地模型（自动检测）
        for model_key, model_path in LOCAL_MODEL_PATHS.items():
            if model_path.exists():
                models[model_key] = {
                    "name": str(model_path),
                    "description": f"本地模型: {model_key}",
                    "size_gb": 1.0,  # 可以根据实际文件大小调整
                    "recommended_vram": 4,
                    "cache_dir": None,
                    "type": "local"
                }
            else:
                print(f"⚠️  本地模型路径不存在: {model_path}")

        # 1. 添加HuggingFace模型
        for key, config in HF_MODELS.items():
            models[key] = config.copy()  # 深拷贝，避免修改原始配置

        return models

    def _check_model_exists(self, model_path_str):
        """检查模型文件是否存在"""
        model_path = Path(model_path_str)

        # 如果是HuggingFace模型名，直接返回True（会在线下载）
        if "/" in model_path_str and not model_path.exists():
            return True

        if not model_path.exists():
            return False

        # 检查是否有必要的模型文件
        required_patterns = [
            "*.bin", "*.safetensors",  # 模型权重
            "config.json", "*.json",  # 配置文件
            "tokenizer.json", "*.model"  # tokenizer文件
        ]

        for pattern in required_patterns:
            if list(model_path.glob(pattern)):
                return True

        return False

    def get_model_info(self, model_key):
        """获取模型信息"""
        if model_key in self.available_models:
            info = self.available_models[model_key]

            # 检查模型状态
            status = "✅ 可用"
            if info.get("type") == "local":
                if not self._check_model_exists(info["name"]):
                    status = "❌ 文件不存在"

            model_path = Path(info["name"]) if info.get("type") == "local" else info["name"]

            return (
                f"📊 模型信息:\n"
                f"• 名称: {model_key}\n"
                f"• 路径: {model_path}\n"
                f"• 状态: {status}\n"
                f"• 描述: {info['description']}\n"
                f"• 大小: {info['size_gb']} GB\n"
                f"• 推荐显存: {info['recommended_vram']} GB\n"
                f"• 类型: {info.get('type', '未知')}"
            )
        return "❌ 未知模型"

    def setup_device(self):
        """设置计算设备"""
        if DML_AVAILABLE:
            import torch_directml
            self.device = torch_directml.device()
            return f"🎮 DirectML设备: {self.device}"
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            return "🎮 CUDA设备"
        else:
            self.device = torch.device("cpu")
            return "⚠️  CPU设备"

    def check_vram_sufficient(self, model_key):
        """检查显存是否足够"""
        if model_key not in self.available_models:
            return False, "未知模型"

        model_info = self.available_models[model_key]
        required_vram = model_info["recommended_vram"]

        if DML_AVAILABLE:
            try:
                import torch_directml
                has_enough = True
                message = f"✅ AMD 7900XTX 24GB显存，可运行{model_key}"
            except:
                has_enough = True
                message = f"⚠️  无法检测DirectML显存，假设足够运行{model_key}"
        elif torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            has_enough = total_vram >= required_vram
            message = f"GPU显存: {total_vram:.1f}GB / 需要: {required_vram}GB"
        else:
            has_enough = False
            message = "❌ 无GPU可用，只能使用CPU运行小模型"

        return has_enough, message

    def load_model(self, model_key):
        """加载指定模型"""
        if model_key not in self.available_models:
            return "❌ 无效的模型选择"

        if self.model_loaded and model_key == self.current_model_key:
            return "✅ 模型已加载，无需重新加载"

        print("=" * 50)
        print(f"🔄 切换模型到: {model_key}")
        print("=" * 50)

        # 检查显存
        has_enough, vram_msg = self.check_vram_sufficient(model_key)
        if not has_enough:
            return f"❌ 显存不足: {vram_msg}"

        model_info = self.available_models[model_key]

        # 检查本地模型文件
        if model_info.get("type") == "local":
            model_path = Path(model_info["name"])
            if not model_path.exists():
                return (
                    f"❌ 本地模型文件不存在\n"
                    f"路径: {model_path}\n\n"
                    f"💡 请将模型文件复制到上述位置\n"
                    f"或选择其他在线模型"
                )

        try:
            # 设置设备
            device_msg = self.setup_device()
            print(device_msg)

            # 如果已有模型，先清理
            if self.model is not None:
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                print("🧹 清理上一个模型")

            # 导入transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # 加载tokenizer
            print(f"🔧 加载tokenizer: {model_info['name']}")

            # 设置缓存目录
            cache_dir = model_info.get("cache_dir")
            if cache_dir and model_info.get("type") == "huggingface":
                print(f"📁 使用缓存目录: {cache_dir}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_info["name"],
                cache_dir=cache_dir,
                trust_remote_code=False,
                local_files_only=(model_info.get("type") == "local")
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"✅ Tokenizer加载成功 (词汇量: {self.tokenizer.vocab_size:,})")

            # 加载模型
            print("🔧 加载模型...")
            dtype = torch.float16 if self.device.type != "cpu" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_info["name"],
                torch_dtype=dtype,
                cache_dir=cache_dir,
                trust_remote_code=True,
                local_files_only=(model_info.get("type") == "local")
            ).to(self.device)

            self.model.eval()
            self.model_loaded = True
            self.current_model_key = model_key
            self.current_model = model_info["name"]

            # 预热模型
            print("🔥 模型预热...")
            self._warmup_model()

            print("=" * 50)
            print("✅ 模型加载完成！")

            model_name_display = model_key
            device_type = "DirectML" if DML_AVAILABLE else "CPU"
            if torch.cuda.is_available():
                device_type = "CUDA"

            message = [
                f"✅ 模型加载成功！",
                f"🤖 当前模型: {model_name_display}",
                f"🎮 运行设备: {device_type}",
                f"📊 词汇量: {self.tokenizer.vocab_size:,}",
                f"💾 模型大小: {model_info['size_gb']} GB",
                f"📁 模型路径: {model_info['name']}",
                f"",
                f"💡 可以开始聊天了！"
            ]

            return "\n".join(message)

        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg

    def _warmup_model(self):
        """预热模型"""
        try:
            warmup_text = "你好"
            inputs = self.tokenizer(warmup_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            print("✅ 模型预热完成")
        except Exception as e:
            print(f"⚠️  预热失败: {e}")

    def generate_response(self, message, history=None):
        """生成回复（包含历史上下文）"""
        if not self.model_loaded:
            return "请先加载模型！", None

        try:
            start_time = time.time()

            messages = []

            if history and len(history) > 0:
                for msg in history[-8:]:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append({
                            "role": msg["role"],
                            "content": str(msg["content"])
                        })

            messages.append({"role": "user", "content": message})

            print(f"📝 发送给模型的消息格式: {messages}")

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            max_tokens = 512
            if "1.5B" in self.current_model_key:
                max_tokens = 384
            elif "3B" in self.current_model_key:
                max_tokens = 512

            input_length = inputs["input_ids"].shape[1]
            max_tokens = min(1024 - input_length, max_tokens)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            gen_time = time.time() - start_time
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            speed = tokens / gen_time if gen_time > 0 else 0

            output = f"{response}\n\n{'=' * 40}\n"
            output += f"📊 生成统计:\n"
            output += f"• 速度: {speed:.1f} token/秒\n"
            output += f"• 长度: {tokens} tokens\n"
            output += f"• 时间: {gen_time:.2f}秒\n"
            output += f"• 上下文长度: {input_length} tokens\n"
            output += f"• 模型: {self.current_model_key}\n"
            output += f"• 类型: {self.available_models[self.current_model_key].get('type', '未知')}\n"
            output += f"{'=' * 40}"

            return output, self.current_model_key

        except Exception as e:
            error_msg = f"❌ 生成失败: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, None


# ————————————————————————————EnhancedAIChat类
class EnhancedAIChat:
    """增强版AI聊天"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.memory_manager = MemoryManager()
        self.qa_process = None
        self.qa_port = 7861

    def get_model_list(self):
        """获取可用模型列表"""
        models = list(self.model_manager.available_models.keys())
        # 按类型排序：本地模型在前，在线模型在后
        local_models = [m for m in models if self.model_manager.available_models[m].get("type") == "local"]
        online_models = [m for m in models if self.model_manager.available_models[m].get("type") == "huggingface"]
        return local_models + online_models

    def get_model_details(self, model_key):
        """获取模型详细信息"""
        return self.model_manager.get_model_info(model_key)

    def switch_model(self, model_key):
        """切换模型"""
        return self.model_manager.load_model(model_key)

    def chat(self, message, history):
        """处理聊天"""
        text_history = []

        if history:
            for msg in history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if isinstance(content, list):
                        text_content = ""
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text_content += item.get("text", "")
                            elif isinstance(item, str):
                                text_content += item

                        if text_content and role in ["user", "assistant"]:
                            text_history.append({"role": role, "content": text_content})
                    elif isinstance(content, str):
                        text_history.append({"role": role, "content": content})

        response, model_used = self.model_manager.generate_response(message, text_history)

        if model_used and "❌" not in response:
            self.memory_manager.add_conversation(message, response, model_used)

        return response, model_used

    def get_memory_summary(self):
        """获取记忆摘要"""
        return self.memory_manager.get_recent_memory(5)

    def clear_memory(self):
        """清空记忆"""
        return self.memory_manager.clear_memory()

    def get_current_model(self):
        """获取当前模型"""
        return self.model_manager.current_model_key

    def launch_qa_interface(self):
        """启动QA界面并返回URL"""
        import subprocess
        import sys
        import os
        from pathlib import Path

        if self.qa_process is not None:
            try:
                self.qa_process.terminate()
                self.qa_process = None
            except:
                pass

        try:
            base_dir = Path(__file__).parent.parent.parent
            qa_script_path = base_dir / "src" / "qa" / "qa_interface.py"

            if not qa_script_path.exists():
                return f"❌ QA脚本不存在: {qa_script_path}"

            print(f"🚀 启动QA界面: {qa_script_path}")
            print(f"📁 项目根目录: {base_dir}")

            env = os.environ.copy()
            python_path = env.get('PYTHONPATH', '')
            if str(base_dir) not in python_path:
                env['PYTHONPATH'] = f"{str(base_dir)}{os.pathsep}{python_path}"
            env['HF_ENDPOINT'] = "https://hf-mirror.com"

            self.qa_process = subprocess.Popen(
                [sys.executable, str(qa_script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(base_dir),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            )

            import time
            time.sleep(3)

            if self.qa_process.poll() is not None:
                stdout, stderr = self.qa_process.communicate()
                error_msg = f"❌ QA进程已退出:\n标准输出:\n{stdout}\n\n标准错误:\n{stderr}"
                print(error_msg)
                return error_msg

            qa_url = f"http://127.0.0.1:{self.qa_port}"
            qa_local_url = f"http://localhost:{self.qa_port}"

            return (f"✅ QA界面已启动！\n"
                    f"📁 工作目录: {base_dir}\n"
                    f"🔗 本地访问: {qa_local_url}\n"
                    f"🌐 网络访问: {qa_url}\n\n"
                    f"💡 点击链接或在浏览器中打开上述地址")

        except Exception as e:
            import traceback
            error_msg = f"❌ 启动QA界面失败: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

    def stop_qa_interface(self):
        """停止QA界面"""
        if self.qa_process is not None:
            try:
                self.qa_process.terminate()
                self.qa_process = None
                return "✅ QA界面已停止"
            except Exception as e:
                return f"❌ 停止QA界面失败: {e}"
        return "ℹ️ 没有运行的QA界面"

# ————————————————————————————界面布局
def create_enhanced_interface():
    """创建增强版界面"""
    ai_chat = EnhancedAIChat()

    print("🔄 创建界面中...")
    print(f"📊 可用模型: {ai_chat.get_model_list()}")

    with gr.Blocks(title="AMD 7900XTX AI助手 - 增强版", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 AMD 7900XTX AI助手 - 增强版")
        gr.Markdown("**模型切换** | **对话记忆** | **DirectML加速**")
        gr.Markdown(f"📁 项目目录: `{PROJECT_ROOT}`")

        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 模型控制")

                model_selector = gr.Dropdown(
                    choices=ai_chat.get_model_list(),
                    value=ai_chat.get_model_list()[0] if ai_chat.get_model_list() else None,
                    label="选择AI模型",
                    interactive=True
                )

                model_info = gr.Textbox(
                    label="📊 模型信息",
                    value=ai_chat.get_model_details(
                        ai_chat.get_model_list()[0]) if ai_chat.get_model_list() else "无可用模型",
                    lines=8,
                    interactive=False
                )

                load_btn = gr.Button("🚀 加载/切换模型", variant="primary")

                status_display = gr.Textbox(
                    label="📈 加载状态",
                    value="请选择模型并点击加载",
                    lines=6,
                    interactive=False
                )

                gr.Markdown("---")
                gr.Markdown("### 🧠 对话记忆")

                memory_display = gr.Textbox(
                    label="📚 记忆历史",
                    value=ai_chat.get_memory_summary(),
                    lines=8,
                    interactive=False
                )

                with gr.Row():
                    refresh_memory_btn = gr.Button("🔄 刷新记忆")
                    clear_memory_btn = gr.Button("🗑️ 清空记忆", variant="stop")

                current_model_display = gr.Textbox(
                    label="🤖 当前模型",
                    value="未加载",
                    interactive=False
                )

                # QA面板
                gr.Markdown("---")
                gr.Markdown("### 📄 文档问答 (QA)")

                qa_status = gr.Textbox(
                    label="QA状态",
                    value="未启动",
                    lines=4,
                    interactive=False
                )

                qa_url_display = gr.Textbox(
                    label="QA访问地址",
                    value="点击启动按钮获取地址",
                    lines=3,
                    interactive=False
                )

                with gr.Row():
                    qa_start_btn = gr.Button("🚀 启动QA界面", variant="secondary")
                    qa_stop_btn = gr.Button("🛑 停止QA", variant="stop")

                with gr.Row():
                    qa_open_btn = gr.Button("🌐 浏览器打开QA界面", variant="primary")

                gr.Markdown("💡 **提示**: QA界面启动后，请复制上方地址到浏览器打开")

            # 右侧聊天区域
            with gr.Column(scale=2):
                gr.Markdown("### 💬 聊天界面")

                chatbot = gr.Chatbot(
                    label="对话",
                    height=550,
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="💭 输入消息",
                        placeholder="请输入您的问题...",
                        lines=3,
                        max_lines=5,
                        scale=4
                    )
                    deep_think_btn = gr.Button(
                        "🤔 深度思考",
                        variant="secondary",
                        scale=1,
                        size="sm",
                        min_width=100
                    )
                    no_deep_think_btn = gr.Button(
                        "🤔 否深度思考",
                        variant="secondary",
                        scale=1,
                        size="sm",
                        min_width=100
                    )

                with gr.Row():
                    send_btn = gr.Button("📤 发送", variant="primary")
                    clear_chat_btn = gr.Button("🗑️ 清空当前对话")

                stats_display = gr.Textbox(
                    label="📊 实时统计",
                    value="等待第一次生成...",
                    interactive=False
                )

        # ===== 事件绑定 =====
        # 深度思考按钮
        def deep_think_prefix(message):
            if not message.strip().startswith('/think'):
                return message + "/think "
            return message

        deep_think_btn.click(
            fn=deep_think_prefix,
            inputs=[msg],
            outputs=[msg]
        )

        def no_deep_think_prefix(message):
            if not message.strip().startswith('/no_think'):
                return message + "/no_think "
            return message

        no_deep_think_btn.click(
            fn=no_deep_think_prefix,
            inputs=[msg],
            outputs=[msg]
        )

        # 1. 模型选择器更新信息
        def update_model_info(model_key):
            return ai_chat.get_model_details(model_key)

        model_selector.change(
            update_model_info,
            inputs=[model_selector],
            outputs=[model_info]
        )

        # 2. 加载/切换模型
        def on_load_model(model_key):
            status = ai_chat.switch_model(model_key)
            current_model = ai_chat.get_current_model() or "未加载"
            return status, current_model

        load_btn.click(
            on_load_model,
            inputs=[model_selector],
            outputs=[status_display, current_model_display]
        )

        # 3. 发送消息
        def on_send_message(message, history, current_model):
            if not message.strip():
                return "", history, "请输入有效消息", current_model

            if current_model == "未加载":
                return "", history, "⚠️ 请先加载模型！", current_model

            try:
                processed_history = []
                if history:
                    for msg in history:
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")

                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        text_parts.append(item.get("text", ""))
                                    elif isinstance(item, str):
                                        text_parts.append(item)

                                if text_parts:
                                    processed_history.append({
                                        "role": role,
                                        "content": " ".join(text_parts)
                                    })
                            elif isinstance(content, str):
                                processed_history.append({"role": role, "content": content})

                response, model_used = ai_chat.chat(message, processed_history)

                if history is None:
                    history = []

                history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message}]
                })

                history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": response}]
                })

                stats_text = "生成完成"
                if "生成统计:" in response:
                    try:
                        stats_lines = response.split("生成统计:")[1].split("\n")
                        stats = [line.strip() for line in stats_lines if line.strip().startswith("•")]
                        if stats:
                            stats_text = " | ".join([s.replace("• ", "") for s in stats[:2]])
                    except:
                        pass

                return "", history, stats_text, current_model

            except Exception as e:
                error_msg = f"发送失败: {str(e)}"
                print(f"❌ 错误: {error_msg}")
                return "", history, error_msg, current_model

        send_btn.click(
            on_send_message,
            inputs=[msg, chatbot, current_model_display],
            outputs=[msg, chatbot, stats_display, current_model_display]
        )

        msg.submit(
            on_send_message,
            inputs=[msg, chatbot, current_model_display],
            outputs=[msg, chatbot, stats_display, current_model_display]
        )

        # 4. 记忆管理
        def refresh_memory():
            return ai_chat.get_memory_summary()

        refresh_memory_btn.click(
            refresh_memory,
            outputs=[memory_display]
        )

        clear_memory_btn.click(
            ai_chat.clear_memory,
            outputs=[memory_display]
        )

        # 5. 清空当前对话
        clear_chat_btn.click(
            lambda: ([], "对话已清空", ai_chat.get_current_model() or "未加载"),
            outputs=[chatbot, stats_display, current_model_display]
        )

        # 6. QA按钮事件
        def start_qa_interface():
            result = ai_chat.launch_qa_interface()
            url = "未获取到URL"
            if "http://" in result:
                import re
                urls = re.findall(r'http://[^\s]+', result)
                if urls:
                    url = urls[0]
            return result, url

        def stop_qa_interface():
            result = ai_chat.stop_qa_interface()
            return result, "已停止"

        def open_qa_browser():
            url = f"http://127.0.0.1:{ai_chat.qa_port}"
            webbrowser.open(url)
            return f"✅ 已尝试打开QA界面: {url}"

        qa_start_btn.click(
            start_qa_interface,
            outputs=[qa_status, qa_url_display]
        )

        qa_stop_btn.click(
            stop_qa_interface,
            outputs=[qa_status, qa_url_display]
        )

        qa_open_btn.click(
            open_qa_browser,
            outputs=[qa_status]
        )

        # 7. 页面加载时初始化
        demo.load(
            lambda: ai_chat.get_memory_summary(),
            outputs=[memory_display]
        )

    return demo

def main():
    print("🌐 启动增强版AI助手...")
    print("💡 新功能:")
    print("  • 支持切换0.5B/1.5B/3B模型")
    print("  • 对话记忆保存与查看")
    print("  • 实时显存检测")
    print("  • 生成统计信息")
    print("-" * 50)
    print(f"💡 本地访问: http://127.0.0.1:{UI_CONFIG['server_port']}")
    print(f"🌐 局域网访问: http://你的IP:{UI_CONFIG['server_port']}")

    demo = create_enhanced_interface()

    demo.launch(
        server_name=UI_CONFIG["server_name"],
        server_port=UI_CONFIG["server_port"],
        share=UI_CONFIG["share"],
        show_error=True,
        debug=False
    )


if __name__ == "__main__":
    main()