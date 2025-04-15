import os
import csv

import pandas as pd
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import argparse
import matplotlib.pyplot as plt

# setting random seed
seed = 42
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_name, model_path, mcq_file):
    if model_name == "DPD_GLM":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device).eval()
        model_output = DPD_GLM_output

    elif model_name == "ChatGLM-4-9B":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
        model_output = ChatGLM4_9B_output

    else:
        raise ValueError(f"Model {model_name} not recognized.")

    print(f"Evaluating {model_name}...")

    df = pd.read_csv(mcq_file)
    merged_df = pd.DataFrame(columns=["question", "response", "answer"])
    prompt = "\n请根据题干和选项，给出唯一的最佳答案。输出内容仅为选项英文字母，不要输出任何其他内容。不要输出汉字。"

    for _, row in tqdm(df.iterrows(), total=len(df)):
        response = model_output(row["question"] + prompt, model, tokenizer)
        merged_df = pd.concat([merged_df, pd.DataFrame({
            "question": [row["question"]],
            "response": [response],
            "answer": [row["answer"]]
        })])

    merged_df['responseTrimmed'] = merged_df['response'].apply(
        lambda x: re.search(r'[ABCDE]', x).group() if re.search(r'[ABCDE]', x) else None
    )
    merged_df['check'] = merged_df.apply(
        lambda row: 1 if row['responseTrimmed'] == row['answer'][0] else 0, axis=1
    )
    score = merged_df["check"].mean()
    merged_df.to_csv(f"src/eval/data/{model_name}_mcq_output.csv", index=False)

    return model_name, score


def ChatGLM4_9B_output(content,model,tokenizer):
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的医生，请基于诊疗指南，为以下患者提供综合的管理意见:",
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    outputs = model.generate(**inputs, max_length=2000, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def DPD_GLM_output(content, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的医生，请基于诊疗指南，为以下患者提供综合的管理意见:",
        },
        {
            "role": "user",
            "content": content,
        },
    ]

    inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

    inputs = inputs.to(device)
    
    
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def save_score(model_name, score, filename='src/eval/results/mcq_scores.csv'):
    records = []
    updated = False

    if os.path.isfile(filename):
        with open(filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader, None)
            for row in reader:
                if row[0] == model_name:
                    records.append([model_name, score])  
                    updated = True
                else:
                    records.append(row)

    if not updated:
        records.append([model_name, score])  # Add new if not found

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model_name', 'score'])
        writer.writerows(records)


def plot_scores_from_csv(csv_file='src/eval/results/mcq_scores.csv', output_dir='src/eval/results', output_name='mcq_benchmark.png'):
    model_scores = {}

    # Read the scores from the CSV
    if not os.path.isfile(csv_file):
        print(f"CSV file '{csv_file}' not found.")
        return

    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            model_scores[row['model_name']] = float(row['score'])

    if not model_scores:
        print("No data to plot.")
        return

    # Prepare data for plotting
    model_names = list(model_scores.keys())
    scores = list(model_scores.values())
    scores_percentage = [score * 100 for score in scores]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, scores_percentage, color="skyblue", width=0.1)

    plt.xlabel("Model Name")
    plt.ylabel("% Correct")
    plt.title("MCQ Benchmark")
    plt.ylim(0, 100)

    for i, score in enumerate(scores_percentage):
        plt.text(i, score + 1, f"{score:.2f}%", ha="center", va="bottom")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    plt.savefig(output_path, format="png")
    plt.show()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="DPD_GLM",
        help="The name of model to be evaluated",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="../autodl-tmp/DPD_GLM",
        help="The path of model to be evaluated",
    )
    args = parser.parse_args()
    
    #model_name, score = evaluate_model(args.model, args.path , "src/eval/data/mcq.csv")
    #save_score(model_name, score)
    plot_scores_from_csv()
