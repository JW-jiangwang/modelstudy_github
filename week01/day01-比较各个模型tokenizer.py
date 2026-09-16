import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer

# 两个不同的 tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

texts = [
    "大模型开发很难吗？",
    "Is large model development hard?",
    "我喜欢吃apple和banana",
    "今天心情很好😊",
]

for text in texts:
    print("=" * 60)
    print(f"原文：{text}")
    print(f"--- Qwen tokenizer ---")
    qwen_tokens = qwen_tokenizer.tokenize(text)
    print(f"token 列表：{qwen_tokens}")
    print(f"token 数量：{len(qwen_tokens)}")
    print(f"--- deepseek tokenizer ---")
    gpt2_tokens = deepseek_tokenizer.tokenize(text)
    print(f"token 列表：{gpt2_tokens}")
    print(f"token 数量：{len(gpt2_tokens)}")