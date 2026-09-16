# ============================================================
# Day 4：用 Chroma 做向量存储和检索
# 对比 Day 3 手写的 VectorStore，理解向量库帮你做了什么
# ============================================================

import os
os.environ["NO_PROXY"] = "hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions


# ------------------------------------------------------------
# 1. 创建 Chroma 客户端（持久化模式）
# ------------------------------------------------------------
# PersistentClient 会把数据存到磁盘，下次启动还在
# 对比 Day 3 的 save/load，Chroma 帮你自动做了这件事
client = chromadb.PersistentClient(path="./chroma_db")


# ------------------------------------------------------------
# 2. 指定 embedding 函数
# ------------------------------------------------------------
# Chroma 默认用 all-MiniLM-L6-v2，对中文支持一般
# 这里换成中文优化的 bge-small-zh
# 注意：Chroma 会自己调用这个模型，你不用手动 encode
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)


# ------------------------------------------------------------
# 3. 创建（或获取）一个 collection
# ------------------------------------------------------------
# collection 类似于数据库里的"表"
# get_or_create：有就获取，没有就创建
collection = client.get_or_create_collection(
    name="my_docs",
    embedding_function=embedding_fn
)


# ------------------------------------------------------------
# 4. 添加文本
# ------------------------------------------------------------
# Chroma 的 add 需要三个参数：
#   documents：文本列表
#   ids：每条文本的唯一 ID（必须唯一）
#   metadatas：可选的元数据（后面做 RAG 时用来存来源、页码等）
collection.add(
    documents=[
        "苹果公司发布了新手机",
        "华为发布了新平板",
        "小米汽车正式上市",
        "今天天气真好",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],
    metadatas=[
        {"source": "tech_news", "date": "2024-09-19"},
        {"source": "tech_news", "date": "2024-09-20"},
        {"source": "auto_news", "date": "2024-03-28"},
        {"source": "weather", "date": "2024-09-21"},
    ]
)

print(f"当前 collection 里有 {collection.count()} 条文档")


# ------------------------------------------------------------
# 5. 检索
# ------------------------------------------------------------
# query：查询文本
# n_results：返回几条
# Chroma 内部自动做：encode → 算相似度 → 排序 → 返回 top_n
results = collection.query(
    query_texts=["我要买苹果手机"],
    n_results=3
)

print("\n=== 检索：'我要买苹果手机' ===")
# results 的结构：{"documents": [[...]], "metadatas": [[...]], "distances": [[...]], "ids": [[...]]}
# 注意是双层列表，因为支持一次查多个 query
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    dist = results["distances"][0][i]
    print(f"距离：{dist:.4f}  {doc}  (来源：{meta['source']})")


# ------------------------------------------------------------
# 6. 再试一个 query
# ------------------------------------------------------------
results2 = collection.query(
    query_texts=["我想买台电动车"],
    n_results=3
)

print("\n=== 检索：'我想买台电动车' ===")
for i in range(len(results2["documents"][0])):
    doc = results2["documents"][0][i]
    dist = results2["distances"][0][i]
    print(f"距离：{dist:.4f}  {doc}")