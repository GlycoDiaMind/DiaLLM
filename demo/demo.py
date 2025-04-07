import sys
import time
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define model path
model_path = "../../autodl-tmp/DiabetesPDiagLLM"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
tokenizer.padding_side="left"
tokenizer.pad_token="[PAD]"
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

def loading_indicator():
    """Displays a loading indicator while waiting for the LLM response."""
    while not stop_loading:
        sys.stdout.write("\rThinking... ")
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write("\r")
    sys.stdout.flush()

def query_llm(prompt):
    global stop_loading
    stop_loading = False
    loader_thread = threading.Thread(target=loading_indicator)
    loader_thread.start()
    """Sends the prompt to the LLM and returns the response."""
    messages = [{
                "role": "system",
                "content": "你是一个专业的医生，请基于诊疗指南，为以下患者提供综合的管理意见:"
            },
            {
                "role": "user",
                "content": prompt
            }
           ]

    inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
    inputs = inputs.to(device)

    gen_kwargs = {"max_length": 300, "do_sample": True, "top_k": 1}
    output = ""
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output =  tokenizer.decode(outputs[0], skip_special_tokens=True)

    stop_loading = True
    loader_thread.join()
    return output

while True:
    user_input = input("\nEnter your prompt (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Bye!")
        break
    response = query_llm(user_input)
    print("\nLLM Response:", response)