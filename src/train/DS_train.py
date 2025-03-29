import os
from datetime import datetime
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
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

# 主函数
def main():
    model_name = "../../../autodl-tmp/DeepSeek-R1-Distill-Qwen-32B"
    output_dir = "../../results/DeepSeek-32B-LoRA"
    new_model_dir = "../../../autodl-tmp/DiabetesPDiagLLM"
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
    print(f"train_dataset:{train_dataset}")
    print(f"val_dataset:{val_dataset}")

    # 加载模型（使用 bf16，全精度无量化）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1  # 推荐设置

    # LoRA 配置
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 训练参数
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        weight_decay=0.001,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
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
        tokenizer=tokenizer,
        args=training_arguments,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=None,
        packing=False
    )

    print("\nTraining ...")
    trainer.train()

    # 合并 LoRA 权重
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    merged_model = PeftModel.from_pretrained(base_model, output_dir).merge_and_unload()

    # 保存模型和 tokenizer
    merged_model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)
    print(f"模型保存至: {new_model_dir}")

if __name__ == "__main__":
    main()