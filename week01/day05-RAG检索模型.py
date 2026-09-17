# ============================================================
# Day 5：第一个完整 RAG
# 用户提问 → 检索相关文档 → 拼 prompt → 送给 LLM → 返回答案
# ============================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv


# ------------------------------------------------------------
# 1. 初始化
# ------------------------------------------------------------
load_dotenv()

client_llm = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 删掉旧的 collection，重新建（避免重复添加）
try:
    chroma_client.delete_collection("my_docs")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="my_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# 添加文档
collection.add(
    documents=[
        "苹果公司于2024年9月发布了iPhone 16，起售价5999元，搭载A18芯片。",
        "华为于2024年9月发布了Mate 70，搭载麒麟9100芯片，支持卫星通信。",
        "小米汽车SU7于2024年3月正式上市，起售价21.59万元，续航最高830公里。",
        "今天北京天气晴，气温18到25度，适合外出。",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"],
    metadatas=[
        {"source": "tech_news", "date": "2024-09-10"},
        {"source": "tech_news", "date": "2024-09-15"},
        {"source": "auto_news", "date": "2024-03-28"},
        {"source": "weather", "date": "2024-09-21"},
    ]
)

print(f"知识库里有 {collection.count()} 条文档")


# ------------------------------------------------------------
# 2. 检索函数
# ------------------------------------------------------------
def retrieve(query, top_k=2):
    """从知识库检索和 query 最相关的 top_k 条文档"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    # 提取文档、来源、距离
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return list(zip(docs, metas, distances))


# ------------------------------------------------------------
# 3. 拼 prompt
# ------------------------------------------------------------
def build_prompt(query, retrieved_docs):
    """把检索到的文档和用户问题拼成 prompt"""
    # 把检索到的文档拼成一段上下文
    context = "\n\n".join([
        f"[{i+1}] {doc}（来源：{meta['source']}，日期：{meta['date']}）"
        for i, (doc, meta, _) in enumerate(retrieved_docs)
    ])

    prompt = f"""你是一个基于文档回答问题的助手。

请根据下面提供的文档回答用户问题。如果文档里没有相关信息，直接说"文档中没有相关信息"，不要编造。

文档：
{context}

用户问题：{query}

回答："""
    return prompt


# ------------------------------------------------------------
# 4. 调用 LLM
# ------------------------------------------------------------
def ask(query, top_k=2):
    """完整的 RAG 流程：检索 → 拼 prompt → 调用 LLM"""
    # 第 1 步：检索
    retrieved = retrieve(query, top_k=top_k)
    print(f"\n=== 检索到 {len(retrieved)} 条相关文档 ===")
    for doc, meta, dist in retrieved:
        print(f"  距离 {dist:.4f}  {doc}")

    # 第 2 步：拼 prompt
    prompt = build_prompt(query, retrieved)

    # 第 3 步：调用 LLM
    response = client_llm.chat.completions.create(
        model="qwen3.7-plus",   # 换成你实际用的模型
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content
    return answer, retrieved


# ------------------------------------------------------------
# 5. 测试
# ------------------------------------------------------------
if __name__ == "__main__":
    questions = [
        "iPhone 16 什么时候发布的？",
        "小米汽车续航多少？",
        "特斯拉 Model 3 多少钱？",   # 知识库里没有，测试模型会不会编
    ]

    for q in questions:
        print("=" * 60)
        print(f"问题：{q}")
        answer, retrieved = ask(q, top_k=2)
        print(f"\n回答：{answer}\n")

