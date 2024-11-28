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


def diabetesPDiagLLM_output(content, model, tokenizer):
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


def main(args):
    model_path = args.path
    if args.model == "DiabetesPDiagLLM":
        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        tokenizer.padding_side="left"
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModel.from_pretrained(model_path,trust_remote_code=True).to(device)
        model_output = diabetesPDiagLLM_output
        
        
    model = model.eval()
    scores = {}
    mcq_list = [
        "src/eval/data/merged_mcq.csv",
    ]

    for i, file_path in enumerate(mcq_list):
        file_name = file_path.split("/")[-1]
        print(f"Processing {file_path} ...")
        df = pd.read_csv(file_path)
        merged_df = pd.DataFrame(columns=["question", "response", "answer"])
        prompt = "\n请根据题干和选项，给出唯一的最佳答案。输出内容仅为选项英文字母，不要输出任何其他内容。不要输出汉字。"

        for index, row in tqdm(df.iterrows()):
            output = {}
            response = model_output(row["question"] + prompt, model, tokenizer)
            output["question"] = [row["question"]]
            output["response"] = [response]
            output["answer"] = row["answer"]
            df_output = pd.DataFrame(output)
            merged_df = pd.concat([merged_df, df_output])

        # calculate MCQ score
        merged_df["responseTrimmed"] = merged_df["response"].apply(
            lambda x: (
                re.search(r"[ABCDE]", x.split("<|assistant|>")[1].split("<|user|>")[0].strip()).group() if re.search(r"[ABCDE]", x.split("<|assistant|>")[1].split("<|user|>")[0].strip()) else None
            )
        )
        merged_df["check"] = merged_df.apply(
            lambda row: 1 if row["responseTrimmed"] == row["answer"][0] else 0, axis=1
        )
        
        score = merged_df["check"].mean()
        name = f"{args.model}"
        merged_df.to_csv("src/eval/data/mcq_with_model_response.csv",index=False)
        scores[name] = score
        
    print(scores)
    
    # Convert the dictionary into two lists: model names and scores
    model_names = list(scores.keys())
    scores = list(scores.values())

    # Convert scores to percentages
    scores_percentage = [score * 100 for score in scores]

    # Plotting the scores
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, scores_percentage, color='skyblue',width=0.4)

    # Adding titles and labels
    plt.xlabel('Model Name')
    plt.ylabel('% Correct')
    plt.title('MCQ Benchmark')
    plt.ylim(0, 100)  # Set y-axis range from 0 to 100

    # Adding the percentage labels on top of the bars
    for i, score in enumerate(scores_percentage):
        plt.text(i, score + 1, f'{score:.2f}%', ha='center', va='bottom')
    
    # Save the plot to a file
    plt.savefig('model_performance.png', format='png')  # Save as PNG file

    # Show the plot
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="DiabetesPDiagLLM",
        help="The name of model to be evaluated",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="../autodl-tmp/fine_tuned_chatglm_for_diabetes",
        help="The path of model to be evaluated",
    )
    args = parser.parse_args()
    main(args)
