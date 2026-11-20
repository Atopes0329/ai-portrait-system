# AI 人像锁脸生成

双击 `启动.command` 即可使用。自动重 roll 由你选的 AI 执行——界面「执行 AI」下拉可选 **Codex / Claude Code / 自定义命令**，生图能力来自所选 AI，不需要 OpenAI API Key。

## 快速开始

### 1. 安装依赖

本项目在**项目虚拟环境 `.venv`** 中运行，不污染系统 Python：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. 下载模型

运行以下脚本自动下载所有必需模型到 `models/` 目录：

```bash
.venv/bin/python download_models.py
```

或者手动下载（见下方"模型下载"章节）。

### 3. 启动程序

- **macOS**: 双击 `启动.command`
- **Windows**: 双击 `启动.bat`
- **命令行**: `.venv/bin/python app.py`（macOS/Linux）或 `.venv\Scripts\python.exe app.py`（Windows）

## 使用流程

1. 选择图片1：本人清晰正脸。程序会记住该路径，以后可以不重复选择。
2. 选系统摄影风格时，可再添加最多 7 张同一人的辅助真人照；图片1始终是唯一身份基准，其余图只补充角度和表情。
3. 需要精确模仿动作和构图时改选图片2；此时辅助真人图自动停用，固定只发送图片1与已遮脸图片2。
4. 调整身份严格度和最多重 roll 次数；默认是严格的 `0.90` 与最多 `6` 次，然后点击“开始锁脸生成”。

系统摄影风格会在本地压缩为最多 8 个画面约束，并自动去掉样例人物的性别、年龄、脸型、五官、发型、眼镜和文字排版要求。小比例人物、背影、群像、倒影、严重遮挡和动漫风会标记为锁脸受限；默认列表只显示锁脸优先风格。

每个任务都保存在 `任务输出/日期时间_任务名/`：

- `图片2_已彻底遮脸.png`：真正送入生图工具的图2；
- `图片2_遮罩审计图.png`：白色区域表示已遮挡范围；
- `参考图分析.json`：从已马赛克图2提取的动作、服装、构图、光线、色调、家具和背景结构；不包含图2人脸身份；
- `candidates/roll_NN.png`：各轮全分辨率候选；
- `candidates/*.score.json`：每轮 SFace、ArcFace、AdaFace 独立分数与图2串脸检查；
- `场景与质量模板.txt`：去掉显式身份块的后续轮干净模板；
- `下一轮提示词.audit.json`：提示词字符预算、身份基线/当前重点计数和警告；
- `最终结果.png`：达标图或状态机在退化回退后保留的历史最佳图；
- `最终报告.json`：是否达标及完整分数。

提示词风格模式不会产生任何图片2文件，而会额外保存 `系统摄影风格.json`，其中只有经过清理的风格名称、最多 8 个画面约束和锁脸兼容性，不包含原始长文档。

## 模型下载

程序需要 4 个 ONNX 运行模型，总大小约 192 MiB。Face Mesh 是可选模型，约 2.4 MiB。所有模型应放在 `models/` 目录下。

### 自动下载（推荐）

```bash
python3 download_models.py
```

### 手动下载

如果自动脚本失败，可以手动下载：

#### 必需模型（共 4 个）

1. **face_detection_yunet_2023mar.onnx** (228 KB)
   https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

2. **face_recognition_sface_2021dec.onnx** (37 MB)
   https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

3. **arcfaceresnet100-11-int8.onnx** (63 MB)
   https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx

4. [adaface_ir_18.onnx](https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_18.onnx)（约 92 MiB）

#### Face Mesh 模型（可选，2.4 MB）

**face_mesh_Nx3x192x192.onnx** 用于可选的468点稠密建议层和校准实验，不是运行硬依赖：

- 原始仓库：https://github.com/yakhyo/uniface （若已不可用，可跳过此模型）

下载后将所有 `.onnx` 文件放入 `models/` 目录。

模型缺失、哈希不匹配、加载失败、姿态不适合二维比较或 presence 低于门槛时，评分报告会记录降级原因并继续使用 YuNet 五点机制。当前稠密指标只参与注意力优先级和排序，不单独触发身份拒绝。

### 模型校验

下载完成后可以运行校验脚本：

```bash
.venv/bin/python verify_models.py
```

该脚本会检查文件是否存在、大小是否合理，以及 SHA-256 是否匹配（见 `models/README.md`）。

## 安全设计

- 图2固定使用约 4×N 的超粗马赛克，保留头部朝向和手脸位置。马赛克后如果检测器仍能识别人脸，程序只会继续加粗马赛克；绝不会退回纯色或不透明遮挡。最强马赛克仍无法消除检测时，任务会安全停止。
- 系统摄影风格模式没有图2，不运行马赛克和图2串脸门；它仍完整运行 P0/P1/P2 身份基线、SFace/ArcFace/AdaFace 身份评分、注意力门和单重点闭环。
- 图2原图路径不会写入 `job.json`，不会传给 Codex 或生图工具。
- 图1与图2原脸只在本机转成数字特征。原图2不会保存进任务目录，也不会交给生图工具。
- 身份评分同时记录 SFace、ArcFace、AdaFace。三者分值尺度不同，因此不做虚假的简单平均：SFace 使用严格门槛，ArcFace 与 AdaFace 用来拦截两个模型都判定明显不符的异常候选。
- 候选必须同时满足：图1身份达到门槛；没有被两个辅助模型共同否定；没有越过图2串脸警戒；人脸尺寸足够评分。
- 身份评分前先做画面完整性检查：解码失败、过小、近乎空图或明显纯黑边会直接重生，不会被误判成“五官不像”。
- 人脸检测会先合并同一张脸上的重复框，再判断单人/多人，原始框数与合并原因保留在审计字段。
- 闭环状态机每轮只修一个身份重点；未达标不切换，整体身份或身份来源失守时优先恢复全局身份。
- 最终候选默认服从闭环状态机保留的历史最佳；真人用 `--prefer` 指定的视觉优选具有最高优先级，状态不可用时才进入保守机器兜底。
- 每轮提示词从干净模板重建，完整测量只留在内部，不随轮次累积到提示词。
- 可选AI视觉评分默认只作旁观建议；没有真人标注校准时不能单独决定通过。
- 身份分只判断"像不像本人"。动作、服饰、构图、滤镜、自然度仍由自动流程做第二层视觉复核。
- 达到最多轮数后一定停止，避免无限消耗生图额度。

## 命令行使用（可选）

使用系统摄影风格建立任务：

```bash
.venv/bin/python face_guard.py prepare \
  --source "/路径/图片1.jpg" \
  --style-preset "<风格文件名>.md" \
  --prompt-file "提示词模板.txt" \
  --threshold 0.90 --max-rolls 6
```

系统风格文件必须来自相邻 `ai人像摄影技能包/ai人像摄影提示词/`。使用图2建立任务：

```bash
.venv/bin/python face_guard.py prepare \
  --source "/路径/图片1.jpg" \
  --style "/路径/图片2.jpg" \
  --prompt "$(cat 提示词模板.txt)" \
  --threshold 0.90 --max-rolls 6
```

给一张候选图评分：

```bash
.venv/bin/python face_guard.py score --job "/路径/job.json" --candidate "/路径/候选.png"
```

推进一次“修一个—验一个”的闭环，并生成下一轮干净提示词：

```bash
.venv/bin/python face_guard.py iterate \
  --job "/路径/job.json" \
  --candidate "/路径/candidates/roll_01.png" \
  --identity-anchor-file "/路径/identity_锁脸段.txt"
```

结果写入任务目录的 `iteration_state.json`；活动状态下同时覆盖写入 `下一轮提示词.txt` 和 `下一轮提示词.audit.json`。后续轮的身份基线与当前重点都只出现一次。生图脚本可直接使用：

```bash
python3 ../ai人像摄影技能包/scripts/generate.py chat \
  --prompt-file "/路径/下一轮提示词.txt" \
  --ref "/路径/图片1_本人.jpg" \
  --ref "/路径/图片2_已彻底遮脸.png"
```

真人阈值校准使用 `calibrate_thresholds.py`。输入CSV至少包含 `human_recognizable` 和一个评分列；样本不足时工具只输出 `uncalibrated`，不会伪造“人类可辨识百分比”。

本程序不会覆盖或修改图片1、图片2原文件。

## Agent 使用

如需在 Claude Code 或其他 AI Agent 中使用，请参阅 `AGENTS.md`。

## 文件结构

```
人脸自动相似替换/
├── README.md              # 本文档
├── requirements.txt       # Python 依赖
├── download_models.py     # 模型下载脚本
├── verify_models.py       # 模型校验脚本
├── 启动.command            # macOS 启动入口
├── app.py                 # GUI 主程序
├── face_guard.py          # 核心遮脸与评分逻辑
├── identity_ensemble.py   # 多模型身份验证
├── model_bakeoff.py       # 模型对比实验工具
├── calibrate_thresholds.py # 真人识别阈值校准
├── identity_lock/          # 模型无关注意力与闭环状态机
├── vision_judges/          # 可替换AI视觉评分提供器
├── 提示词模板.txt          # 生图提示词模板
├── AGENTS.md              # Agent 自动重 roll 规则
├── models/                # 模型目录（需下载）
│   ├── README.md          # 模型来源与 SHA-256
│   └── *.onnx             # 4 个运行模型及可选 Face Mesh
└── 任务输出/               # 所有任务输出保存在此
```

## 许可证与引用

第三方预训练模型的来源与许可证见 [模型说明](models/README.md)。

用于商业客户项目前请再次核对上游模型的训练数据与使用条款。
