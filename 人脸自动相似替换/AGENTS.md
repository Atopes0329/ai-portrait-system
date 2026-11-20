# 人脸自动相似替换：自动重 roll 规则

当任务含 `AUTO_PORTRAIT_REROLL` 时，严格执行以下流程。

> 环境约定：本技能包所有 Python 命令从技能包根目录运行，统一使用虚拟环境解释器（macOS/Linux：`.venv/bin/python`；Windows：`.venv\Scripts\python.exe`）。依赖装在 `.venv` 内，不依赖系统 Python。

1. 读取用户指定的 `job.json`、首轮 `提示词.txt` 和后续轮 `场景与质量模板.txt`，先判断 `job.mode`。第二轮起只使用系统生成的 `下一轮提示词.txt`，并检查同名 `.audit.json` 中的重复计数与预算警告；不得在其后再手工累加完整身份档案。
2. 分模式处理参考图：
   - `portrait`：没有图2，跳过 `reference_analysis_file` 与遮脸分析；使用 `generation_reference_images` 中全部真人照片。若 `style_source.kind=prompt_library`，系统摄影风格已经压缩写入 `提示词.txt`，只使用该精简段，禁止寻找或读取原始风格 Markdown。
   - `replacement`：只查看已经马赛克的 `masked_style_image`，把动作、服装、构图、镜头、光线、色调、家具和背景写入 `参考图分析.json` 的 `semantic_analysis`，并把状态改为 `complete`。不得描述、猜测或复原马赛克区域中的脸。
3. 参考图规则：
   - `portrait`：按 `generation_reference_images` 顺序全部传入；第一张是唯一基准，其余只补充真人多角度信息。系统摄影风格是文本约束，不增加第二张人物图。
   - `replacement`：只把 `source_image` 与 `masked_style_image` 作为生图参考图。图片1是唯一身份源，图片2只提供动作/服饰/滤镜。
4. 绝对禁止寻找、读取或向生图工具传入图2原图。`job.json` 不保存原图2路径；`private/identity_features.npz` 只允许交给 `face_guard.py` 评分，不得读取、解释或传给任何生图工具。
5. **图1 特征建模（每次任务只做一次，图1不变则复用）**：用视觉模型把图片1的面部特征翻译成文字，写进每轮生图提示词做「特征文字锁」。这是对抗"生图模型对参考图零样本理解"的关键——锁脸越稳，重 roll 越少。
   - 复用 `ai人像摄影技能包` 的 `vision.py`（视觉接口配置读该技能包的 `config.json`，若该文件未配置，先让使用者按 ai人像摄影技能包 的首次配置流程配好 vision 接口）。下面路径是相对「人脸自动相似替换」目录的相对路径，两个 skill 在同一父目录（如分发盒子）时有效，否则按实际位置改：
     ```
     .venv/bin/python "../ai人像摄影技能包/scripts/vision.py" <job.json 的 source_image> --out <job.json 的 session_dir>/identity_锁脸段.txt
     ```
   - 产出的 `identity_锁脸段.txt` 内含完整内部档案和「基线锁脸段成品」。**首轮传入经 P0/P1/P2 分级的身份基线；P3 不传。评分后每轮固定基线，只额外允许一个活动重点。**
   - 上轮低分时只读取状态机的 `current_focus`，每轮只允许一个活动身份重点。
   - 区分两个身份文件，不要混淆：
     - `private/identity_features.npz` = 本地数字特征，**只允许交给 `face_guard.py` 评分**，不得读取、解释或传给任何生图工具（沿用第 4 条）。
     - `identity_锁脸段.txt` = 视觉模型生成的完整文字档案；只提取其中「基线锁脸段成品」用于提示词，其余只供内部选择重点。
6. 每轮只生成一张候选图，必须直接落盘到 `candidates/roll_NN`。`portrait` 使用全部 `generation_reference_images`；`replacement` 使用图片1和已遮脸图片2。从 `参考图分析.json` 取约束只适用于 replacement，且最多 6～8 个。**提示词优先级固定为：P0/P1/P2身份基线 > 真人参考图 > 当前一个修正重点 > 场景/动作 > 灯光滤镜。** 如果动作与身份冲突，改变动作而不是改变脸。并追加：
   - `prompt_library` 的风格段已经去除样例人物身份、发型、眼镜和文字要求；不得把 `style_source.preset_title` 扩写回完整样例人物描述，也不得在后续轮次累积更多风格词。
   - 当前是第 N 轮；
   - 上轮低分时，只使用 `current_focus.instruction` 指定的一个身份重点；
   - 上轮疑似串脸时，再次强调不得猜测图片2遮挡区。
7. 将每轮全分辨率结果保存到 `candidates/roll_NN.png`，不得只留聊天预览。如果工具返回落盘提示，按该提示复制；不得截图或二次压缩。
8. 每轮生成后运行闭环评分入口：

   `.venv/bin/python face_guard.py iterate --job <job.json> --candidate <roll_NN.png> --identity-anchor-file <identity_锁脸段.txt>`

   若已配置独立AI视觉评分，再加 `--vision-config <config.json>`；未配置或接口失败时自动降级为本地评分。

9. 读取对应的 `.score.json`：
   - `gates.quality=false`：候选图解码、尺寸、空图或纯黑边检查失败，本轮不进入身份比较；直接重新生成完整画面，不得把它当成五官偏差去改脸。
   - `passed=true` 只是机器身份门槛，不得立即当成最终结论；继续做一次肉眼视觉复核；
   - `passed=false`：依据 `reject_reasons` 调整下一轮提示；
   - 到 `max_rolls` 仍未达标：停止生成，不得无限消耗额度。
   - **必读 `attention_priority`（身份扫描路径控制器），它是正式验收与下轮重点的依据**：
     - `decision=reject`：候选图存在可靠测得的身份锚点错误；必须重 roll，且只使用状态机选出的一个 `current_focus`。不得重新堆叠全部五官描述。
     - `decision=unreliable`：当前脸太小、姿态/透视过强或参考档案不完整，不能据此说“不是本人”，但本轮也不能作为锁脸通过；改构图、脸部尺寸或角度后再生成可评估候选。
     - `decision=pass`：已校准硬锚点通过。若 `advisory_repair_recommended=false`，保持现有锁脸段；若为 `true` 且还有轮数，只允许额外生成一轮按 `next_focus` 定向修正的候选，保留原通过图一起排序，不得把原图说成“身份失败”。
     - `instruction` 是模型无关的下一轮修正指令；不得自行把低优先级皮肤、情绪、服装混入身份修正预算。
     - `evidence_mode=advisory` 的五点嘴部/Face Mesh 稠密指标可以进入 `next_focus`，但不能单独作为“不是本人”的理由。
     - `key=identity_source_conflict` 只会出现在 replacement 模式；先加强“图1唯一身份、不得继承图2脸”，不要把问题误归因到单个五官。
   - **必读 `iteration_decision`**：`retry_focus` 继续当前重点；`advance_focus` 才能换下一个；`retry_observability` 先修构图；`complete` 进入最终复核；`stop` 停止消耗额度并保留历史最佳图。
   - `vision_judge.gate_mode=advisory_shadow` 时，AI分只记录不放行；只有真人样本校准完成并显式启用后才可参与退化拦截，且仍不能覆盖本地三模型。
10. 肉眼复核必须把候选与图片1并排检查：真实年龄感、脸宽/脸长、颊部体积、眼距和眼型、鼻宽鼻长、嘴型笑容、下颌轮廓。`replacement` 再依据 `参考图分析.json` 检查图片2；`prompt_library` 则依据 `系统摄影风格.json` 检查精简画面要素，不检查也不报告图2分数。任何明显漂移都继续重抽，即使机器门槛已过线。SFace、ArcFace、AdaFace 是身份护栏，不是最终审美裁判。
11. 最后运行：

    `.venv/bin/python face_guard.py finalize --job <job.json>`

    如果肉眼最优图不是状态机保留图，使用：

    `.venv/bin/python face_guard.py finalize --job <job.json> --prefer <肉眼最优候选图>`

    未传 `--prefer` 时，`finalize` 必须优先采用 `iteration_state.json` 已保留的历史最佳候选；只有状态文件缺失、损坏或候选没有评分记录时，才能使用保守机器兜底，并在最终报告记录降级原因。不得用另一套临时排序覆盖状态机的退化回退。`--prefer` 也不能绕过画面完整性与可评分性硬门。

12. 确认 `最终结果.*` 与 `最终报告.json` 存在后才报告完成；只有 `replacement` 还必须确认完成状态的 `参考图分析.json`。明确说明是否真正达标、实际轮数、图1相似度和最终文件路径；图2相似度只在 replacement 模式报告，prompt_library 改为报告系统风格名称与兼容性。

不得修改模型、门槛或任务 JSON 来让失败结果"通过"。不得用后期直接贴图1原脸来伪造重 roll 分数，除非用户另行明确要求做本地换脸后期。
