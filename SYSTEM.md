# AI人像完整技能包 · 系统架构

本包的核心是一套 **身份锁定系统**：判断候选图是否仍是图1中的同一个人，并把有限的修正预算优先给最关键的身份锚点。AI 写真 / 海报 / 杂志封面是目前附带的一种调用场景，不是核心机制本身。

---

## 一、系统组成

### 核心：人脸自动相似替换（身份扫描路径控制器 + 本地验脸）
| 部件 | 职责 |
|---|---|
| `identity_lock/` | **模型无关的身份扫描路径控制器**：建立图1的最小身份档案，判断可观测性与偏差，并以单重点状态机控制“修一个—验一个”；不依赖 Gemini 或任意生图 API |
| `landmark_providers/` | 可选稠密 landmark 适配层；Face Mesh 可用时补充脸型、眼区、鼻部与条件性嘴部指标，不可用时自动降级 |
| `face_guard.py` | 双风格来源任务 + 遮脸 + 本地三模型评分 + 身份锚点验收 + 闭环推进；AI视觉评分可选，缺失时仍可纯本地运行 |
| `candidate_quality.py` | 身份评分前的画面完整性硬门：拦截解码失败、过小、近乎空图和明显纯黑边 |
| `face_detection_utils.py` | 合并 YuNet 对同一张脸的重复检测框，保留原始/唯一人脸数审计 |
| `identity_lock/prompt_budget.py` | 从无身份场景模板重建后续轮次，保证身份基线和当前重点各一次，并输出字符预算审计 |
| `style_library.py` | 本地读取摄影风格 Markdown，去除样例人物身份与文字要求，压缩为最多 8 个画面约束并标记锁脸兼容性 |
| `vision_judges/` | 可替换AI视觉评分；校准前只旁观记录，不绑定任何生图模型 |
| `IDENTITY_CORE.md` | 核心机制、适用边界与校准要求 |
| `models/` | 4 个 ONNX 模型（初始化下载并校验 SHA-256） |

### 调用层 1：ai人像摄影技能包（锁脸提示词引擎）
| 部件 | 职责 |
|---|---|
| `scripts/vision.py` | **特征建模**：输出完整内部档案和首轮 P0/P1/P2 身份基线；支持档案防覆盖、自检补全、多角度、模型降级与大图压缩 |
| `scripts/profile_guard.py` | 管理 provisional / validated 状态、档案哈希和 regression-only 黄金成片 |
| `scripts/generate.py` | 生图：`chat`（nano-banana 锁脸最强）/ `draw`（gpt-image）/ `export`（手动粘平台）三模式 |
| `ai人像摄影提示词/` | 88 个 Markdown：87 个可选摄影/视觉风格，另有 1 个身份还原说明文档（不会出现在风格列表） |
| `SKILL.md` | 给 AI 的主流程手册（建模→三段式→生图→迭代） |

### 调用层 2：自动评分重 roll（可选）
| 部件 | 职责 |
|---|---|
| `app.py` | GUI 入口，调 codex / claude CLI 自动跑重 roll |
| `AGENTS.md` | 给 AI 的自动重 roll 规则（含图1特征建模步骤） |

### 支撑件（本包根目录）
`首次配置问卷.md`（AI 引导配置）· `SYSTEM.md`（本文）· `README.md`（快速开始）· `setup.py` / `初始化.command` / `初始化.bat`（一键初始化）· `分发清单.txt`

---

## 二、协同流程（一次完整出图）

```
[首次] 问卷配置 → config.json
   │
   ↓
① 档案检查    validated 直接复用；新人物建立 provisional 档案
   ↓
② 首轮基线    P0/P1/P2 分级锁脸段 + 全部有效真人参考图（图1 + 最多7张辅助图）
   ↓
③ 选择风格    prompt_library（系统精简提示词）/ image_reference（遮脸图2）
   ↓
④ 建立任务    face_guard.py prepare（portrait / replacement）
   ↓
⑤ 生图评分    generate.py 落盘候选 → face_guard.py iterate
   ↓
⑥ 完整性门  先排除坏图/黑边，再进入身份评分
   ↓
⑦ 迭代        未达标留在当前重点；达标后才移动；每轮重建干净提示词并写预算审计
```

关系：**身份扫描路径控制器负责验明和优先分配；人像摄影技能包负责建模/出图；自动重 roll 是把两者串起来的可选执行器。** 核心本身可被其他任意模型调用；不使用外部模型时，它仍能完成本地检测、身份评分、可观测性判断与候选排序，但不会自行生成图像。

---

## 三、配置依赖图

```
config.json（ai人像摄影技能包/ 根目录）
 ├─ vision.*        ← vision.py 读它做特征建模
 │                   ← 人脸自动相似替换 的锁脸段（identity_锁脸段.txt）也读它
 ├─ vision_judge.*  ← 可选AI视觉评分；未校准时只作影子建议
 └─ generate.*      ← generate.py 读它生图

人脸自动相似替换/.venv   ← face_guard.py / app.py 的依赖（numpy/opencv/onnxruntime）
人脸自动相似替换/models/  ← 评分模型（4 个 onnx）
                              可选第 5 个 Face Mesh 仅增强稠密建议层
```

**关键依赖**：人脸自动相似替换 的「图1特征建模」步骤依赖子系统 1 的 `vision.py` 和它的 `config.json`。**两个 skill 必须放在同一父目录下**（本包就是这个形态），否则相对路径引用断裂。

---

## 四、环境约定

| 环境 | 说明 |
|---|---|
| ai人像摄影技能包 | **纯 Python 标准库**（无第三方依赖），用系统 `python3` 跑 |
| 人脸自动相似替换 | 依赖 numpy/opencv/onnxruntime，装在 **`.venv`** |
| 跨 skill 调用 | AGENTS.md 用 `.venv` 的 python 跑 vision.py（vision.py 无依赖也能跑，统一用 `.venv` 避免混淆） |

### 平台差异（重要）
| 操作 | macOS / Linux | Windows |
|---|---|---|
| 虚拟环境 python | `.venv/bin/python` | `.venv\Scripts\python.exe` |
| 一键初始化 | 双击 `初始化.command` | 双击 `初始化.bat` |
| 启动 GUI | 双击 `启动.command` | 双击 `启动.bat` |
| 手动初始化 | `python3 setup.py` | `python setup.py` |

---

## 五、跨平台支持

- **macOS**：双击 `初始化.command` / `启动.command`
- **Windows**：双击 `初始化.bat` / `启动.bat`（setup.py 自动识别平台）
- **Linux**：`python3 setup.py`，之后 `.venv/bin/python app.py`

初始化（setup.py）自动做：校验目录关系 → 检查 Python 3.8+ → 生成 config 空白模板 → 建 .venv 装依赖 → 校验/补下载模型 → 校验 4 个运行模型的 SHA256 并实际加载 ONNX Runtime → 打印当前平台正确命令。任一运行模型未就绪时会明确失败，不会把不可启动的 GUI 标成“初始化完成”。

---

## 六、已知边界（诚实说明）

1. **自动重 roll 必须有 AI agent**（Codex 或 Claude Code CLI 之一）。遮脸、评分是本地的；"看脸分析、拼提示词、每轮调整"这层智能必须有 agent 执行。纯手动出图不需要。
2. **生图效果跟选的模型有关**。问卷推荐组合（vision=gemini-2.5-pro；生图 nano-banana-2 或 gpt-image-2）效果最稳；换模型会有差异。
3. **模型已打包**（约 194MB），换来对方免下载 + 效果一致。
4. **大图压缩**：vision.py 优先 PIL → opencv（.venv 自带）→ macOS sips，三选一都能压；Windows 用 .venv 跑就有 opencv，能压缩。
5. **隐私**：`config.json` 含 API 密钥，绝不分发；系统提示词模式发送图1与用户明确添加的最多 7 张辅助真人图，图2模式则固定只发送图1与已遮脸图2，图2原图绝不外传。自动重 roll 默认在完成后清理任务内的复制件、身份向量、身份提示词和候选图，只保留最终图、报告和日志。
6. **Face Mesh 是可选增强，不是硬依赖**：当前分发包没有附带该实验模型时，系统会明确记录 `missing` 并继续使用 YuNet 五点、SFace、ArcFace、AdaFace；不会假装已经启用468点判断。
7. **人类可辨识阈值必须校准**：原始模型分不是百分比。没有足量真人“能认出/不能认出”标注时，AI视觉分只能旁观记录，最终仍需人工确认。

---

## 七、目录约定

```
ai人像摄影系统/            ← 系统根（两个 skill + 支撑件必须在同一父目录）
├── AGENTS.md / SYSTEM.md / README.md / 分发清单.txt / 首次配置问卷.md / 迭代进化.md
├── setup.py / 初始化.command / 初始化.bat
├── ai人像摄影技能包/（运行时本地建立 特征库/；干净分发时为空）
└── 人脸自动相似替换/
```

改源后生成干净分发：运行 `python3 package_clean.py "/一个尚不存在的新目录"`。脚本不覆盖旧目录，并强制排除人物档案、照片、成片、任务输出、密钥和个人迭代记录。

---

## 八、自我进化机制（系统越用越好）

本系统带「家书式」自我进化：**`迭代进化.md`（系统根目录）是它的记忆**，随系统一起分发。

- 每次出图任务**开始前**：AI 读 `迭代进化.md`——查「人物特征索引」直接复用已建模特征（不重复建模），扫「经验条目」避开已知坑。
- 每次任务**收尾后**：AI 按「经验条目」格式追加一条（做了什么 / 效果 / 哪些有效 / 教训 / 下次改进）。
- 新人物特征存 `ai人像摄影技能包/特征库/<人物>_特征.md`，并登记进迭代日志。

使用者用得越多，系统越懂 TA 的审美和常用人物——这就是「越用越好」。
