# 用于工作流调用

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer_32b():
    load_directory = "../../../autodl-tmp/DiabetesPDiagLLM"

    tokenizer = AutoTokenizer.from_pretrained(load_directory, trust_remote_code=True)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        load_directory,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    return model, tokenizer

def inference(user_content: str, model, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "你是一个专业的医生，请基于诊疗指南，为以下患者提供综合的管理意见:"},
        {"role": "user", "content": user_content}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            top_k=1
        )

    response = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response

# 测试入口
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    sample_input = "患者空腹血糖升高，餐后血糖波动较大，伴有轻微视力模糊。"
    result = inference(sample_input, model, tokenizer)
    print("模型输出结果：\n", result)