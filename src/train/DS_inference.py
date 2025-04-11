# Author: MangguoD
# DS_inference.py for workflow
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer_32b(model_path: str = None):
    """
    加载 DeepSeek-32B 模型及对应的 tokenizer。
    参数:
      model_path: 模型本地路径目录。如未指定则使用默认路径。
    返回:
      (model, tokenizer) 元组。
    """
    if model_path is None:
        model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../autodl-tmp/DiabetesPDiagLLM"))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"[错误] 模型路径不存在：{model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
#    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True
    ).eval()
    
    return model, tokenizer

# 输出处理
def process_response(text):
    think_tag = "</think>"
    if think_tag in text:
        return text.split(think_tag)[-1].strip()
    return text.strip()

# 推理部分
def inference(user_content: str, model, tokenizer) -> str:
    """
    使用 DeepSeek-32B 模型对用户输入的病情描述进行推理，
    返回模型生成的结果文本（清洗后的）。
    """
    messages = [
        {"role": "system", "content": (
            "你是一个专业的医生，你的任务是："
            "1.风险评估：根据患者病情病例，输出相应的医疗风险评估。"
            "2.健康建议：结合患者的年龄、近期血糖控制情况、血压等并发症情况、用药与依从性情况等，给出专业的健康建议。"
            "3.饮食运动建议：根据患者 BMI 以及体检情况计算每日建议摄入的卡路里，结合患者体质给出每日运动相关建议。"
            "请基于诊疗指南，为以下患者提供综合的管理意见:"
        )},
        {"role": "user", "content": user_content}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

#    print("[原始输出]:\n", response)
    return process_response(response)


# 测试入口（仅作本模块独立运行时测试使用）
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer_32b()
    sample_input = "在这里输入病情。"
    result = inference(sample_input, model, tokenizer)
    print("模型输出结果：\n", result)