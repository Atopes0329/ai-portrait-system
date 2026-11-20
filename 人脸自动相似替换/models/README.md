# 本地模型来源

这些 ONNX 模型在本机运行。初始化会从上游下载 4 个运行模型并校验 SHA-256。手动下载后，将文件放入此目录并运行 `python3 verify_models.py`。

| 文件 | 用途 | 官方来源 | 许可证 | SHA-256 |
| --- | --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 人脸检测与五点定位 | [OpenCV Zoo / YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | 主身份门槛与图2串脸检查 | [OpenCV Zoo / SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache-2.0 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| `arcfaceresnet100-11-int8.onnx` | 身份交叉检查 | [ONNX Model Zoo / ArcFace](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface) | Apache-2.0 | `c625ca68a422418c48aa84f73341337e0a92b111f327909005d1eec07c95f936` |
| `adaface_ir_18.onnx` | 画质变化下的身份交叉检查 | [AdaFace ONNX 上游权重](https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_18.onnx) | MIT | `6b6a35772fb636cdd4fa86520c1a259d0c41472a76f70f802b351837a00d9870` |
| `face_mesh_Nx3x192x192.onnx` | 可选468点稠密建议层与离线校准 | [UniFace / MediaPipe Face Mesh ONNX](https://github.com/yakhyo/uniface) | Apache-2.0 | `3ca77cf59c18e4da0eccb46695bf604683fa564253e3385892981a5c274fb10f` |

注意：代码许可证与训练数据、预训练权重的使用条件可能不同；用于商业客户项目前应再次核对上游条款。

Face Mesh 当前不是运行硬依赖。只有文件大小与 SHA256 均匹配、模型可加载、姿态适合且 presence 足够时才启用；其指标在完成真实样本校准前只作为建议证据。
