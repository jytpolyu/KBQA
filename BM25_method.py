import math
import time
import logging
from collections import Counter

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bm25_keyword_search.log"),
        logging.StreamHandler()
    ]
)

def compute_tf(document):
    """
    计算文档的词频 (TF)
    :param document: list, 分词后的文档
    :return: dict, 每个词的词频
    """
    try:
        tf = Counter(document)
        total_terms = len(document)
        if total_terms == 0:
            logging.warning("文档为空，无法计算词频")
            return {}
        for term in tf:
            tf[term] /= total_terms
        return tf
    except Exception as e:
        logging.error(f"计算词频时发生错误: {e}")
        raise

def compute_idf(documents):
    """
    计算逆文档频率 (IDF)
    :param documents: list of list, 每个文档是分词后的词列表
    :return: dict, 每个词的 IDF 值
    """
    try:
        num_documents = len(documents)
        if num_documents == 0:
            logging.warning("文档列表为空，无法计算 IDF")
            return {}
        idf = {}
        all_terms = set(term for doc in documents for term in doc)
        for term in all_terms:
            containing_docs = sum(1 for doc in documents if term in doc)
            idf[term] = math.log((num_documents - containing_docs + 0.5) / (containing_docs + 0.5) + 1)
        return idf
    except Exception as e:
        logging.error(f"计算 IDF 时发生错误: {e}")
        raise

def compute_bm25_score(tf, idf, doc_length, avg_doc_length, k1=1.5, b=0.75):
    """
    计算 BM25 分数
    :param tf: dict, 词频
    :param idf: dict, 逆文档频率
    :param doc_length: int, 文档长度
    :param avg_doc_length: float, 文档平均长度
    :param k1: float, 调节参数
    :param b: float, 调节参数
    :return: float, BM25 分数
    """
    try:
        score = 0.0
        for term, term_tf in tf.items():
            term_idf = idf.get(term, 0)
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += term_idf * (numerator / denominator)
        return score
    except Exception as e:
        logging.error(f"计算 BM25 分数时发生错误: {e}")
        raise

def bm25_keyword_search(documents, query):
    """
    使用 BM25 方法进行关键词匹配
    :param documents: list of dict, 文档列表，每个文档包含 'document_id' 和 'document_text'
    :param query: str, 查询字符串
    :return: list of tuple, 包含文档 ID 和 BM25 分数的元组列表，按分数降序排序
    """
    try:
        start_time = time.time()  # 开始计时

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

        # 计算 IDF
        idf = compute_idf(tokenized_documents)

        # 计算文档平均长度
        avg_doc_length = sum(len(doc) for doc in tokenized_documents) / len(tokenized_documents)

        # 计算每个文档的 BM25 分数
        scores = []
        for doc, doc_id in zip(tokenized_documents, documents_ids):
            tf = compute_tf(doc)
            score = compute_bm25_score(tf, idf, len(doc), avg_doc_length)
            scores.append((doc_id, score))

        # 按分数降序排序
        ranked_results = sorted(scores, key=lambda x: x[1], reverse=True)

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        logging.info(f"BM25 匹配完成，共处理 {len(documents)} 篇文档")
        return ranked_results, elapsed_time
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

    results = bm25_keyword_search(documents, query)

    # 输出结果
    for doc_id, score in results:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
"""