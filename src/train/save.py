# Author: MangguoD
# 用于保存模型

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 请确保这两个路径都已经存在：
# base_model_ckpt: 存放原始基础模型的路径或 checkpoint
# adapter_ckpt: 保存 adapter 的文件夹（应包含 adapter_config.json 和 pytorch_model.bin）
base_model_ckpt = "../../../autodl-tmp/DeepSeek-R1-Distill-Qwen-32B"
adapter_ckpt = "DeepSeek-32B-LoRA-SumDataset"  # adapter checkpoint 保存的目录（相对于当前工作目录）

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_ckpt,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 加载 adapter（离线模式下，确保 adapter_ckpt 目录下有完整的保存文件）
peft_model = PeftModel.from_pretrained(
    base_model,
    adapter_ckpt,
    local_files_only=True
)

# 合并 adapter 权重到基础模型
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型和 tokenizer
save_dir = "../../../autodl-tmp/DiabetesPDiagLLM"
merged_model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_ckpt, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

print(f"模型已保存至: {save_dir}")