# Author: MangguoD

import os
from datetime import datetime
import torch
import torch.nn as nn
import gc
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer,SFTConfig
from accelerate import Accelerator


def load_data(data_files: str,tokenizer ) -> DatasetDict:
    def format_chat_template(row):
        row_json = [{"role": row["messages"][0]["role"], "content": row["messages"][0]["content"]},
                    {"role": row["messages"][1]["role"], "content": row["messages"][1]["content"]},
                    {"role": row["messages"][2]["role"], "content": row["messages"][2]["content"]},
                ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row
    
    dataset = load_dataset("json", data_files=data_files)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    train_dataset = train_dataset.map(format_chat_template,num_proc=6)
    val_dataset = val_dataset.map(format_chat_template,num_proc=6)
    train_dataset = train_dataset.remove_columns("messages")
    val_dataset = val_dataset.remove_columns("messages")
    return train_dataset, val_dataset

def inference():
    load_directory="../autodl-tmp/DiabetesPDiagLLM"
    tokenizer = AutoTokenizer.from_pretrained(load_directory,trust_remote_code=True)
    tokenizer.padding_side="left"
    tokenizer.pad_token="[PAD]"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    
    messages = [{
                "role": "system",
                "content": "你是一个专业的医生，你的任务是：1.风险评估：根据患者病情病例，输出相应的医疗风险评估。2.健康建议：结合患者的年龄、近期血糖控制情况、血压等并发症情况、用药与依从性情况等，给出专业的健康建议。3.饮食运动建议：根据患者 BMI 以及体检情况计算每日建议摄入的卡路里，结合患者体质给出每日运动相关建议。请基于诊疗指南，为以下患者提供综合的管理意见:"
            },
            {
                "role": "user",
                "content":"你好，我是一位21岁的男性，每日保持三餐，每天午睡并且晚上有充足的睡眠，一周锻炼1-2次，身高185cm，体重73kg，有点喜欢喝甜甜的饮料，血压正常，暂无依赖性药物，身体免疫力一般，请问我是否有糖尿病的风险，并给出我具体建议。"
            }
           ]

    inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

    inputs = inputs.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        load_directory,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    

def main():
    model_name = "../autodl-tmp/DiabetesPDiagLLM"

    # dataset
    data_files = {
        'train': "data/WeDoctor/train.json",
        'validation': "data/WeDoctor/val.json"
    }

    # Fine-tuned model name
    new_model = "DiabetesPDiagLLM"

    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.6

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    output_dir = "./results"

    # Number of training epochs
    num_train_epochs = 2

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training 原来是4
    per_device_train_batch_size = 1

    # Batch size per GPU for evaluation 原来是4
    per_device_eval_batch_size = 1

    # Number of update steps to accumulate the gradients for 原来是1
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer) 原来是2e-4
    learning_rate = 3e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 0

    # Log every X updates steps
    logging_steps = 10

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"
    
    # load dataset
    train_dataset,val_dataset = load_data(data_files,tokenizer)
    print(f"train_dataset:{train_dataset}")
    print(f"val_dataset:{val_dataset}")
    
    
    
    
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1



    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Modify the trainer initialization to include the train_dataset and eval_dataset
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # Use the split training dataset
        eval_dataset=val_dataset,  # Use the split validation dataset
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model

    # 打印模型字典状态，调试使用
    # print(model.state_dict().keys())
    print("\nTraining ...")
    trainer.train()
    
    # Save model
    trainer.model.save_pretrained(new_model)
    
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True, 
        # bf16
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    #tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = f"../autodl-tmp/DiabetesPDiagLLM"

    # Save the model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
   


    
    
    
if __name__ == "__main__":
    #main()
    inference()