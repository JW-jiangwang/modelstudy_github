import os
os.environ["NO_PROXY"] = "hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

sentences = [
    "今天天气真好",
    "阳光明媚的一天",
    "我喜欢吃苹果",
    "苹果公司发布了新手机",
    "大模型开发很难吗",
]

vectors = model.encode(sentences)
print(f"向量形状：{vectors.shape}")

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n=== '今天天气真好' 和其他句子的相似度 ===")
for i, s in enumerate(sentences):
    sim = cos_sim(vectors[0], vectors[i])
    print(f"{sim:.4f}  {s}")


print("\n\n\n")
#todo 加句话，看看相似度
def test01(query):
    vectors01=model.encode([query])
    score=[]
    for i, s in enumerate(sentences):
        sim = cos_sim(vectors01[0], vectors[i])
        score.append((sim,s))
    score.sort(reverse=True)
    for sim,s in score:
        print(f"{sim:.4f}  {s}与{query}的相似度")
test01("我想吃水果")

print("\n\n\n")
#todo 再加句话，看看相似度
def test02(query):
    vectors01=model.encode([query])
    score=[]
    for i, s in enumerate(sentences):
        sim = cos_sim(vectors01[0], vectors[i])
        score.append((sim,s))
    score.sort(reverse=True)
    for sim,s in score:
        print(f"{sim:.4f}  {s}与{query}的相似度")
test02("苹果18pro9月19发售")
