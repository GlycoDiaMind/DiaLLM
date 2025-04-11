# Author: MangguoD

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 指定 TensorBoard 日志文件目录
log_dir = "DeepSeek-32B-LoRA-SumDataset/runs/Apr09_17-45-43_autodl-container-c22545b252-9d186e6f"
event_file = None

# 查找 .tfevents 文件
for file in os.listdir(log_dir):
    if file.startswith("events.out.tfevents"):
        event_file = os.path.join(log_dir, file)
        break

if event_file is None:
    raise FileNotFoundError("未在日志目录中找到事件文件（.tfevents）")

# 加载事件文件
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 获取所有 scalar tag
available_tags = ea.Tags().get("scalars", [])
print(f"可用的 scalar tags: {available_tags}")

# 要提取的目标 tags
target_tags = [
    "train/loss",
    "train/grad_norm",
    "train/learning_rate"
]

# 遍历每个 tag 并绘图保存
for tag in target_tags:
    if tag not in available_tags:
        print(f"[跳过] 未找到 tag: {tag}")
        continue

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=tag, linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Training Curve: {tag}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    filename = f"{tag.replace('/', '_')}_curve.png"
    output_path = os.path.join(log_dir, filename)
    plt.savefig(output_path)
    plt.close()

    print(f"图像已保存为: {output_path}")