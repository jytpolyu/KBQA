import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tfidf_keyword_search.log"),
        logging.StreamHandler()
    ]
)

def tfidf_keyword_search(documents, query, top_k=1):
    """
    使用 sklearn 的 TfidfVectorizer 和 cosine_similarity 实现关键词匹配
    :param documents: list of dict, 文档列表，每个文档包含 'document_id' 和 'document_text'
    :param query: str, 查询字符串
    :param top_k: int, 控制返回的结果数量，默认为 None 表示返回所有结果
    :return: list, 包含排序后的文档 ID 列表
    """
    try:
        start_time = time.time()  # 开始计时

        # 提取文档文本和文档 ID
        document_texts = [doc.get('document_text', '').lower() for doc in documents]
        document_ids = [doc.get('document_id', -1) for doc in documents]

        if not document_texts:
            logging.warning("文档列表为空，无法进行关键词匹配")
            return []

        if not query.strip():
            logging.warning("查询字符串为空，无法进行关键词匹配")
            return []

        # 使用 TfidfVectorizer 计算 TF-IDF 矩阵
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(document_texts)

        # 将查询转换为 TF-IDF 向量
        query_vector = vectorizer.transform([query.lower()])

        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # 将文档 ID 和相似度配对
        results = list(zip(document_ids, similarities))

        # 按相似度降序排序
        ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

        # 如果指定了 top_k，则只返回前 top_k 个结果
        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        # 提取文档 ID 列表
        ranked_ids = [doc_id for doc_id, _ in ranked_results]

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        logging.info(f"关键词匹配完成，共处理 {len(documents)} 篇文档，耗时 {elapsed_time:.4f} 秒")
        return ranked_ids
    except Exception as e:
        logging.error(f"关键词匹配时发生错误: {e}")
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

    results = tfidf_keyword_search(documents, query, 3)

    # 输出结果
    print(results)
"""
