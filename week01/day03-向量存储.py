# ============================================================
# Day 3：手写一个最简 VectorStore
# ============================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # 加载 embedding 模型
        self.model = SentenceTransformer(model_name)
        # 存原始文本
        self.texts = []
        # 存向量，形状 (N, 512)
        self.vectors = None

    def add(self, text):
        # encode 返回 (1, 512)，取 [0] 变成 (512,)
        vec = self.model.encode([text])[0]
        self.texts.append(text)
        # 第一条时初始化成 (1, 512)，后续 vstack 堆成 (N+1, 512)
        if self.vectors is None:
            self.vectors = vec.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vec])

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query])[0]
        # 批量余弦相似度：一次算出 query 和所有向量的相似度
        scores = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )
        # argsort 返回从小到大索引，[::-1] 反转成从大到小
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            results.append((float(scores[i]), self.texts[i]))
        return results

    def save(self, path):
        # numpy 数组不能直接 json 序列化，要 tolist()
        data = {
            "texts": self.texts,
            "vectors": self.vectors.tolist() if self.vectors is not None else []
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        # list 转回 numpy 数组
        if data["vectors"]:
            self.vectors = np.array(data["vectors"])
        else:
            self.vectors = None


if __name__ == "__main__":
    store = VectorStore()

    store.add("今天天气真好")
    store.add("阳光明媚的一天")
    store.add("我喜欢吃苹果")
    store.add("苹果公司发布了新手机")
    store.add("大模型开发很难吗")

    print(f"存储了 {len(store.texts)} 条文本")
    print(f"向量形状：{store.vectors.shape}")

    print("\n=== 检索：'我想吃水果' ===")
    for sim, text in store.search("我想吃水果", top_k=3):
        print(f"{sim:.4f}  {text}")

    print("\n=== 检索：'苹果18pro9月19发售' ===")
    for sim, text in store.search("苹果18pro9月19发售", top_k=3):
        print(f"{sim:.4f}  {text}")

    store.save("vector_store.json")
    print("\n已保存到 vector_store.json")

    store2 = VectorStore()
    store2.load("vector_store.json")
    print(f"加载后：{len(store2.texts)} 条文本")
    print("\n=== 加载后检索：'我想吃水果' ===")
    for sim, text in store2.search("我想吃水果", top_k=3):
        print(f"{sim:.4f}  {text}")