import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download

GEN_MODEL_NAME = "qwen/Qwen2.5-7B-Instruct"
print(f"Downloading/Loading {GEN_MODEL_NAME}...")
model_dir = snapshot_download(GEN_MODEL_NAME)
print(f"Model dir: {model_dir}")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
print("Model loaded successfully!")
