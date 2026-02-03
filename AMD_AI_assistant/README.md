# 🚀 AMD 7900XTX AI助手

一个基于Qwen模型、使用DirectML加速的本地AI助手，专为AMD GPU优化。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ 特性

- 🎮 **AMD GPU优化**：使用DirectML加速，充分发挥7900XTX性能
- 🤖 **多模型支持**：支持Qwen2.5系列0.5B/1.5B/3B模型
- 💾 **对话记忆**：自动保存对话历史，支持查看和清理
- 📄 **文档问答**：集成文档QA系统，支持本地文档处理
- 🎨 **友好界面**：基于Gradio的Web界面，美观易用
- 🔄 **实时切换**：无需重启即可切换不同模型

## 🚀 快速开始

### 1. 环境安装
```bash
# 克隆项目
git clone https://github.com/yourusername/AMD_AI_Assistant.git
cd AMD_AI_Assistant

# 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt