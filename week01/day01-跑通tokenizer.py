import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



from transformers import AutoTokenizer


# 对应你 Ollama 里的 qwen2:0.5b
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

texts = [
    "大模型开发很难吗？",
    "Is large model development hard?",
    "我喜欢吃apple和banana",
    "今天心情很好😊",
]

for text in texts:
    print("=" * 50)
    print(f"原文：{text}")
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"token 列表：{tokens}")
    print(f"token 数量：{len(tokens)}")
    print(f"token ID：{ids}")
    print(f"解码还原：{tokenizer.decode(ids)}")



