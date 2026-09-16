# ============================================================
# Day 3：手写一个最简 VectorStore
# 目标：封装 embedding + 检索 + 持久化，理解向量库内部原理
# ============================================================

import os
os.environ["NO_PROXY"] = "hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """一个最简的向量存储：能添加文本、检索相似文本、保存到磁盘"""

    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # 加载 embedding 模型，所有文本都通过它转成向量
        self.model = SentenceTransformer(model_name)

        # 存储所有文本（原始字符串）
        self.texts = []

        # 存储所有向量，形状 (N, 512)
        # 用 np.array 而不是 list，方便后面做矩阵运算
        self.vectors = None

    def add(self, text):
        """添加一条文本，自动转成向量存起来"""
        # 把文本转成向量，形状 (1, 512)，取 [0] 变成 (512,)
        vec = self.model.encode([text])[0]

        # 文本加入列表
        self.texts.append(text)

        # 向量加入数组
        if self.vectors is None:
            # 第一条：直接变成 (1, 512)
            self.vectors = vec.reshape(1, -1)
        else:
            # 后续：垂直堆叠，变成 (N+1, 512)
            self.vectors = np.vstack([self.vectors, vec])

    def search(self, query, top_k=3):
        """检索和 query 最相似的 top_k 条文本"""
        # query 也转成向量
        query_vec = self.model.encode([query])[0]

        # 算 query 和所有存储向量的相似度
        # 注意：这里用了 numpy 的批量运算，比 for 循环快
        # 分子：query_vec 和每个向量的点积
        # 分母：query_vec 的模长 × 每个向量的模长
        scores = np.dot(self.vectors, query_vec) / (
            np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        )

        # argsort 返回从小到大排序的索引，[::-1] 反转成从大到小
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 返回 (相似度, 文本) 的列表
        results = []
        for i in top_indices:
            results.append((float(scores[i]), self.texts[i]))
        return results

    def save(self, path):
        """把文本和向量保存到磁盘"""
        data = {
            "texts": self.texts,
            # numpy 数组不能直接 json 序列化，转成 list
            "vectors": self.vectors.tolist() if self.vectors is not None else []
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path):
        """从磁盘加载文本和向量"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.texts = data["texts"]
        if data["vectors"]:
            self.vectors = np.array(data["vectors"])
        else:
            self.vectors = None


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    store = VectorStore()

    # 添加几条文本
    store.add("今天天气真好")
    store.add("阳光明媚的一天")
    store.add("我喜欢吃苹果")
    store.add("苹果公司发布了新手机")
    store.add("大模型开发很难吗")

    print(f"存储了 {len(store.texts)} 条文本")
    print(f"向量形状：{store.vectors.shape}")

    # 检索
    print("\n=== 检索：'我想吃水果' ===")
    for sim, text in store.search("我想吃水果", top_k=3):
        print(f"{sim:.4f}  {text}")

    print("\n=== 检索：'苹果18pro9月19发售' ===")
    for sim, text in store.search("苹果18pro9月19发售", top_k=3):
        print(f"{sim:.4f}  {text}")

    # 保存到磁盘
    store.save("vector_store.json")
    print("\n已保存到 vector_store.json")

    # 新建一个 store，从磁盘加载
    store2 = VectorStore()
    store2.load("vector_store.json")
    print(f"加载后：{len(store2.texts)} 条文本")
    print("\n=== 加载后检索：'我想吃水果' ===")
    for sim, text in store2.search("我想吃水果", top_k=3):
        print(f"{sim:.4f}  {text}")

    store3 = VectorStore()
    store3.load("vector_store.json")
    print(f"加载后：{len(store3.texts)} 条文本")
    print("\n=== 加载后检索：'我要买苹果手机' ===")
    for sim, text in store2.search("我要买苹果手机", top_k=3):
        print(f"{sim:.4f}  {text}")