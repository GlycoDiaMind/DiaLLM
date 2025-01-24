import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

device = "cuda"

def main():
    inference()

def download_llm():
    snapshot_download("ZhipuAI/glm-4-9b-chat", cache_dir="../../autodl-tmp")

def inference():
    tokenizer = AutoTokenizer.from_pretrained("../../autodl-tmp/ZhipuAI/glm-4-9b-chat", trust_remote_code=True)

    query = "Hello"
    
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True
                                           )
    
    inputs = inputs.to(device)
    model = AutoModelForCausalLM.from_pretrained(
        "../../autodl-tmp/ZhipuAI/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
if __name__ == "__main__":
    main()