import os
import torch
import torch.nn as nn
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
from trl import SFTTrainer
from accelerate import Accelerator


@dataclass
class MyTrainingArguments(TrainingArguments):
    """
    Add custom arguments to the existing TrainingArguments class
    """

    lora_rank: int = field(default=8, metadata={"help": "Lora Rank"})
    lora_dropout: float = field(default=0.1, metadata={"help": "Lora Dropout"})
