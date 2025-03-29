import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def inference():
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

    messages = [
        {"role": "system", "content": "你是一个专业的医生，请基于诊疗指南，为以下患者提供综合的管理意见:"},
        {"role": "user", "content": "你好"}
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=True, top_k=1)
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response)

if __name__ == "__main__":
    inference()