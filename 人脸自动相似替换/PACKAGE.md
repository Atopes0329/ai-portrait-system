# 打包清单

获取源码后，先运行根目录的一键初始化脚本安装依赖并下载模型。

## 包含文件

### 核心程序
- `app.py` - GUI 主程序
- `face_guard.py` - 遮脸与评分核心逻辑
- `style_library.py` - 系统摄影风格读取、去身份压缩与锁脸兼容性分级
- `identity_ensemble.py` - 多模型身份验证
- `identity_lock/` - 模型无关的身份注意力核心与稠密比例计算
- `landmark_providers/` - 可选 Face Mesh ONNX 适配层
- `model_bakeoff.py` - 真实样片指标校准工具
- `启动.command` - macOS 一键启动脚本

### 配置与文档
- `README.md` - 完整使用说明
- `AGENTS.md` - AI Agent 使用规则
- `requirements.txt` - Python 依赖清单
- `提示词模板.txt` - 生图提示词模板
- `models/README.md` - 模型来源与校验信息

### 辅助脚本
- `download_models.py` - 自动下载模型
- `verify_models.py` - 验证模型完整性

## 模型说明

初始化会下载 4 个运行模型并校验 SHA-256。Face Mesh 是可选的第5个增强模型；缺失时系统继续使用五点身份锚点，不影响GUI启动。

## 不包含的个人数据
- `任务输出/` - 历史任务结果
- `实验/` - 实验数据
- `本地设置.json` - 个人图片路径记忆
- `参考图分析_当前样片.json` - 特定案例数据
- `多模型擂台_本案例.json` - 特定案例数据
- `辅助检测结论.md` - 特定案例笔记

这些文件包含个人路径或测试数据，不应分享给其他用户。

## 使用步骤

1. 解压到任意目录
2. 安装依赖：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. 下载模型：`.venv/bin/python download_models.py`
4. 启动程序并在“系统摄影风格 / 上传风格参考图”中二选一：
   - macOS: 双击 `启动.command`
   - 其他系统: `.venv/bin/python app.py`

详细说明请查看 `README.md`。
