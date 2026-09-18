# ============================================================
# Day 7：对比两个 embedding 模型的检索效果
# ============================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NO_PROXY"] = "hf-mirror.com"

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions


# ------------------------------------------------------------
# 1. 准备两个 embedding 模型
# ------------------------------------------------------------
# 两个都是中文优化模型，但规格不同：
#   bge-small-zh-v1.5：512 维，小、快
#   bge-base-zh-v1.5：768 维，大、慢、准
# 用同一个数据集，对比检索结果
models = {
    "small": "BAAI/bge-small-zh-v1.5",
    "base": "BAAI/bge-base-zh-v1.5",
}


# ------------------------------------------------------------
# 2. 测试数据
# ------------------------------------------------------------
documents = [
    "苹果公司于2024年9月发布了iPhone 16，起售价5999元，搭载A18芯片。",
    "华为于2024年9月发布了Mate 70，搭载麒麟9100芯片，支持卫星通信。",
    "小米汽车SU7于2024年3月正式上市，起售价21.59万元，续航最高830公里。",
    "今天北京天气晴，气温18到25度，适合外出。",
    "大模型开发需要掌握 Python、Transformer、RAG 等技术。",
]

metadatas = [
    {"source": "tech_news", "date": "2024-09-10"},
    {"source": "tech_news", "date": "2024-09-15"},
    {"source": "auto_news", "date": "2024-03-28"},
    {"source": "weather", "date": "2024-09-21"},
    {"source": "tech_blog", "date": "2024-09-01"},
]

ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

queries = [
    "iPhone 16 什么时候发布的？",
    "小米汽车续航多少？",
    "怎么学大模型开发？",
    "今天天气怎么样？",
]


# ------------------------------------------------------------
# 3. 对每个模型，建一个独立的 collection，跑所有 query
# ------------------------------------------------------------
for model_name, model_path in models.items():
    print("=" * 60)
    print(f"模型：{model_name} ({model_path})")
    print("=" * 60)

    # 用独立的 Chroma 目录，避免两个模型的数据混在一起
    client = chromadb.PersistentClient(path=f"./chroma_db_{model_name}")

    # 删掉旧的 collection，重新建
    try:
        client.delete_collection("test_docs")
    except:
        pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_path
    )

    collection = client.get_or_create_collection(
        name="test_docs",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # 跑每个 query
    for query in queries:
        results = collection.query(query_texts=[query], n_results=3)
        print(f"\n  查询：{query}")
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            dist = results["distances"][0][i]
            print(f"    距离 {dist:.4f}  {doc[:40]}...")