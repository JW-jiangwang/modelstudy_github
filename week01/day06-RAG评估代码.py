# ============================================================
# Day 6：给 RAG 加评估
# 用测试集量化 RAG 的表现
# ============================================================

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------
# 1. 初始化（和 Day 5 一样）
# ------------------------------------------------------------
client_llm = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

try:
    chroma_client.delete_collection("my_docs")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="my_docs",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

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


# ------------------------------------------------------------
# 2. 检索和生成函数
# ------------------------------------------------------------
def retrieve(query, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)
    return list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ))


def build_prompt(query, retrieved_docs):
    context = "\n\n".join([
        f"[{i+1}] {doc}（来源：{meta['source']}）"
        for i, (doc, meta, _) in enumerate(retrieved_docs)
    ])
    return f"""你是一个基于文档回答问题的助手。

请根据下面提供的文档回答用户问题。如果文档里没有相关信息，直接说"文档中没有相关信息"，不要编造。

文档：
{context}

用户问题：{query}

回答："""


def ask(query, top_k=2):
    retrieved = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, retrieved)
    response = client_llm.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    answer = response.choices[0].message.content
    return answer, retrieved


# ------------------------------------------------------------
# 3. 测试集
# ------------------------------------------------------------
# 每个问题包含：
#   query：问题
#   expected_keywords：正确答案必须包含的关键词
#   should_refuse：文档里没有相关信息时，模型应该拒绝回答
test_set = [
    {
        "query": "iPhone 16 什么时候发布的？",
        "expected_keywords": ["2024", "9月"],
        "should_refuse": False
    },
    {
        "query": "iPhone 16 多少钱？",
        "expected_keywords": ["5999"],
        "should_refuse": False
    },
    {
        "query": "iPhone 16 用什么芯片？",
        "expected_keywords": ["A18"],
        "should_refuse": False
    },
    {
        "query": "华为 Mate 70 什么时候发布的？",
        "expected_keywords": ["2024", "9月"],
        "should_refuse": False
    },
    {
        "query": "华为 Mate 70 支持卫星通信吗？",
        "expected_keywords": ["支持", "卫星"],
        "should_refuse": False
    },
    {
        "query": "小米汽车续航多少？",
        "expected_keywords": ["830"],
        "should_refuse": False
    },
    {
        "query": "小米汽车多少钱？",
        "expected_keywords": ["21.59"],
        "should_refuse": False
    },
    {
        "query": "今天北京天气怎么样？",
        "expected_keywords": ["晴"],
        "should_refuse": False
    },
    {
        "query": "特斯拉 Model 3 多少钱？",
        "expected_keywords": [],
        "should_refuse": True
    },
    {
        "query": "苹果公司CEO是谁？",
        "expected_keywords": [],
        "should_refuse": True
    },
]


# ------------------------------------------------------------
# 4. 评估函数
# ------------------------------------------------------------
def evaluate():
    total = len(test_set)
    retrieval_hits = 0       # 检索是否命中（top_k 里有相关文档）
    answer_correct = 0       # 答案是否包含期望关键词
    refuse_correct = 0       # 是否正确拒绝（should_refuse=True 时）
    refuse_total = 0         # 应该拒绝的问题总数

    results_log = []

    for i, case in enumerate(test_set):
        query = case["query"]
        expected = case["expected_keywords"]
        should_refuse = case["should_refuse"]

        print(f"\n[{i+1}/{total}] {query}")

        answer, retrieved = ask(query, top_k=2)

        # 检查检索命中：top 1 的文档是否包含期望关键词
        top1_doc = retrieved[0][0]
        retrieval_hit = any(kw in top1_doc for kw in expected) if expected else False
        if retrieval_hit:
            retrieval_hits += 1

        # 检查答案
        answer_ok = False
        if should_refuse:
            refuse_total += 1
            # 模型应该拒绝
            if "没有" in answer or "未找到" in answer or "无法" in answer:
                refuse_correct += 1
                answer_ok = True
        else:
            # 答案应包含期望关键词
            if all(kw in answer for kw in expected):
                answer_correct += 1
                answer_ok = True

        print(f"  检索 top1 距离：{retrieved[0][2]:.4f}")
        print(f"  检索命中：{'✅' if retrieval_hit else '❌'}")
        print(f"  回答：{answer}")
        print(f"  答案正确：{'✅' if answer_ok else '❌'}")

        results_log.append({
            "query": query,
            "answer": answer,
            "retrieval_hit": retrieval_hit,
            "answer_ok": answer_ok,
            "top1_distance": retrieved[0][2]
        })

    # 汇总
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总问题数：{total}")
    print(f"检索命中率：{retrieval_hits}/{total - refuse_total} = {retrieval_hits/(total-refuse_total)*100:.1f}%")
    print(f"答案准确率：{answer_correct}/{total - refuse_total} = {answer_correct/(total-refuse_total)*100:.1f}%")
    print(f"拒绝编造率：{refuse_correct}/{refuse_total} = {refuse_correct/refuse_total*100:.1f}%")

    # 保存评估报告
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print("\n评估报告已保存到 eval_report.json")


if __name__ == "__main__":
    evaluate()