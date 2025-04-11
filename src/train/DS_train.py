# Author: MangguoD

import os
# 切换为离线模式，并启用可扩展内存分配，降低内存碎片问题
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from accelerate import Accelerator

# 加载数据集
def load_data(data_files: dict, tokenizer) -> DatasetDict:
    def format_chat_template(row):
        row_json = [
            {"role": row["messages"][0]["role"], "content": row["messages"][0]["content"]},
            {"role": row["messages"][1]["role"], "content": row["messages"][1]["content"]},
            {"role": row["messages"][2]["role"], "content": row["messages"][2]["content"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = load_dataset("json", data_files=data_files)
    train_dataset = dataset["train"].map(format_chat_template, num_proc=6).remove_columns("messages")
    val_dataset = dataset["validation"].map(format_chat_template, num_proc=6).remove_columns("messages")
    return train_dataset, val_dataset

# 对文本进行 token 截断
def truncate_text(example, tokenizer, max_tokens=10000):
    tokenized = tokenizer(example["text"], add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    if len(input_ids) > max_tokens:
        truncated_ids = input_ids[:max_tokens]
        example["text"] = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return example

# 自定义学习率调度器函数
def get_custom_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int, initial_lr: float, min_lr: float = 1e-5):
    """
    返回一个 LambdaLR 调度器，使用 Cosine 调度，并确保学习率不会低于 min_lr。
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        min_lr_factor = min_lr / initial_lr
        return cosine_decay * (1 - min_lr_factor) + min_lr_factor
    return LambdaLR(optimizer, lr_lambda)

def main():
    model_name = "../../../autodl-tmp/DeepSeek-R1-Distill-Qwen-32B"
    output_dir = "DeepSeek-32B-LoRA-SumDataset"  # 请确保名称仅包含字母、数字、横杠或下划线
    new_model_dir = "../../../autodl-tmp/DiabetesPDiagLLM-Sum"
    data_files = {
        'train': "../../data/WeDoctor/train.json",
        'validation': "../../data/WeDoctor/val.json"
    }

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    # 加载数据集
    train_dataset, val_dataset = load_data(data_files, tokenizer)
    print(f"train_dataset: {train_dataset}")
    print(f"val_dataset: {val_dataset}")

    # 对数据集中的文本进行截断
    train_dataset = train_dataset.map(lambda ex: truncate_text(ex, tokenizer, max_tokens=10000), num_proc=6)
    val_dataset = val_dataset.map(lambda ex: truncate_text(ex, tokenizer, max_tokens=10000), num_proc=6)

    # 加载模型（使用 bf16，全精度无量化）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # LoRA 配置
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 训练参数设置，初始学习率设为 1e-4
    initial_lr = 1e-4
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        # 上次结构ep=3
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=initial_lr,
        max_grad_norm=0.3,
        weight_decay=0.001,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",  # 内置调度器参数，会覆盖为自定义调度器
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=0,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_arguments,
        peft_config=peft_config,
    )

    # 计算总训练步数和 warmup 步数
    num_train_examples = len(train_dataset)
    effective_batch_size = training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps
    num_training_steps = math.ceil(num_train_examples / effective_batch_size) * training_arguments.num_train_epochs
    num_warmup_steps = int(training_arguments.warmup_ratio * num_training_steps)
    print(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

    # 创建优化器和默认调度器，再用自定义调度器覆盖默认的 lr_scheduler
    trainer.create_optimizer_and_scheduler(num_training_steps=num_training_steps)
    trainer.lr_scheduler = get_custom_scheduler(trainer.optimizer, num_warmup_steps, num_training_steps, initial_lr, min_lr=1e-6)

    print("\nTraining ...")
    trainer.train()  # 开始训练

    # 保存 adapter（完整保存，包含配置和权重）
    trainer.model.save_pretrained(output_dir)
    
    # 释放 GPU 占用以减少内存碎片
    del trainer
    torch.cuda.empty_cache()

    # 重新加载基础模型用于合并 adapter 权重
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    # 将基础模型移到 CPU 进行合并，减少 GPU 内存占用
    base_model = base_model.cpu()
    torch.cuda.empty_cache()

    # 使用相对路径加载 adapter 权重并合并 LoRA 权重
    adapter_path = output_dir
    merged_model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True
    ).merge_and_unload()

    # 保存合并后的模型和 tokenizer
    merged_model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)
    print(f"模型保存至: {new_model_dir}")

if __name__ == "__main__":
    main()