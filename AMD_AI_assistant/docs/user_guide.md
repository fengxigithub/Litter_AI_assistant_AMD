# AMD AI 项目使用说明

## 🚀 启动方式

### 方法1：一键启动（推荐）

### 方法2：Python主菜单
```bash
python main.py
```

### 方法3：直接启动Web
```bash
python src\web\app.py
```

### 方法4：命令行聊天
```bash
python src\core\chat_engine.py
```

## ⚙️ 配置文件

- 模型配置: `config\model_config.json`
- Web配置: 在代码中修改

## 🔧 性能测试

运行性能测试：
```bash
python tests\benchmark.py
```

## 📁 目录说明

- `src\core\` - 核心AI功能
- `src\web\`  - Web界面
- `src\utils\` - 工具函数
- `scripts\`  - 启动脚本
- `tests\`    - 测试文件
- `config\`   - 配置文件
- `archive\`  - 归档文件（旧版本）

## ❓ 常见问题

1. Web界面无法启动？

2. 模型加载失败？

3. GPU加速无效？
  检查DirectML环境是否正确安装

