# 说明：
# 本文件实现了基于 BM25 算法的关键词检索功能。
# 功能模块：
# 1. 文档分词：对文档和查询进行分词处理。
# 2. BM25 分数计算：使用 `rank_bm25` 库计算文档与查询的相关性分数。
# 3. 文档排序：根据 BM25 分数对文档进行排序，并返回前 top_k 个结果。
import time
import logging
from rank_bm25 import BM25Okapi

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bm25_keyword_search.log"),
        logging.StreamHandler()
    ]
)

def bm25_keyword_search(documents, query, top_k=1):
    """
    使用 rank_bm25 库实现 BM25 方法进行关键词匹配
    :param documents: list of dict, 文档列表，每个文档包含 'document_id' 和 'document_text'
    :param query: str, 查询字符串
    :param top_k: int, 控制返回的结果数量，默认为 None 表示返回所有结果
    :return: list, 包含排序后的文档 ID 列表
    """
    try:
        start_time = time.time()  # 开始计时

        print("开始 BM25 分词...")
        # 对文档和查询进行分词
        tokenized_documents = [doc.get('document_text', '').lower().split() for doc in documents]
        documents_ids = [doc.get('document_id', -1) for doc in documents]
        tokenized_query = query.lower().split()

        if not tokenized_documents:
            logging.warning("文档列表为空，无法进行关键词匹配")
            return []

        if not tokenized_query:
            logging.warning("查询字符串为空，无法进行关键词匹配")
            return []

        # 使用 rank_bm25 计算 BM25 分数
        print("开始 BM25 计算分数...")
        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(tokenized_query)

        # 将文档 ID 和分数配对
        results = list(zip(documents_ids, scores))

        print("开始 BM25 分数排序...")
        # 按分数降序排序
        ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

        # 如果指定了 top_k，则只返回前 top_k 个结果
        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        # 提取文档 ID 列表
        ranked_ids = [doc_id for doc_id, _ in ranked_results]

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        logging.info(f"BM25 匹配完成，共处理 {len(documents)} 篇文档，耗时 {elapsed_time:.4f} 秒")
        return ranked_ids
    except Exception as e:
        logging.error(f"BM25 匹配时发生错误: {e}")
        raise

# 示例调用
"""
if __name__ == "__main__":
    documents = [
        {"document_id": 6205, "document_text": "what did the huns do to the roman empire"},
        {"document_id": 8985, "document_text": "who won women's singles australian open 2018"},
        {"document_id": 9541, "document_text": "who plays the gunslinger in the dark tower"}
    ]
    query = "who"

    results = bm25_keyword_search(documents, query, 40)

    print(results)
"""