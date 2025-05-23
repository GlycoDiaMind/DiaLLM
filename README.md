#  展示 Demo 与模型训练说明文档

本说明文档介绍了该项目的 Demo 展示方式、训练流程、代码结构与关键模块功能。

---

##  展示 Demo 操作指南

1. **启动前准备**：
   - 确保预训练模型路径已正确设置；
   - 确认提示词内容符合预期；
   - 默认最大句子长度为 **10000 tokens**；

2. **注意事项**：
   - 请特别注意 **推理过程中的内存释放**，以避免显存溢出；
   - 推荐在 GPU 显存充足的设备上运行。

---

##  简要训练流程

```bash
# 激活虚拟环境
conda activate sum_env

# 进入训练代码目录
cd DiabetesPDiagLLM/

# 执行训练脚本
python src/train/train.py
```

- 请先在 `utils.py` 中完成 **数据预处理**；
- 再启动训练脚本进行模型训练。

---

##  项目功能模块说明

该库是一个完整的医疗文本分析模型训练与展示系统，包含以下功能模块：

| 功能 | 说明 |
|------|------|
| 模型训练 | 支持 DeepSeek-32B 模型微调训练 |
| 推理调用 | 封装独立推理函数用于生产部署 |
| Demo 展示 | 模拟用户输入的前端展示界面 |
| 能力比较 | 可视化不同模型输出效果进行评估 |

---

##  模块与目录说明

### `demo/` 文件夹

- 存放用于模拟用户直接输入的交互式展示脚本；

---

### `src/train/` 目录介绍

#### `DS_train.py`
- 主要训练脚本；
- 使用 DeepSeek-32B 模型进行微调训练；
- 输出模型权重保存至指定目录。

#### `Tensorboard.py`
- 调用 `Tensorboard` 对训练过程进行图形化展示；
- 每次训练输出图像默认以 `.png` 格式保存至根目录。

#### `save.py`
- 本身`train.py`具有保存模型功能，该代码仅在`train`保存失败，显存溢出等情况使用；
- 每次训练结果保存在`base_model_ckpt`目录下。
  
#### `DS_inference.py`
- 提供模型加载与推理函数；
- 为工作流模块提供支持，保证模型可独立初始化调用；
- 与训练流程解耦，便于后续部署与封装。
- 与 Gradio 或 CLI 环境连接进行实时推理。

---

如需进一步展示或集成 Gradio 界面，请确保推理模块接口标准化（如：支持 `inference(text, model, tokenizer)`）。

如有路径配置、虚拟环境或模块结构问题，请联系开发者获取支持。

## Author

[MangguoD](https://github.com/MangguoD)