import math
import time
import logging
from collections import Counter
import numpy as np


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
    handlers=[
        logging.FileHandler("tfidf_keyword_search.log"),  # 输出到文件
        logging.StreamHandler()  # 同时输出到控制台
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
            idf[term] = math.log((num_documents + 1) / (containing_docs + 1)) + 1  # 平滑处理
        return idf
    except Exception as e:
        logging.error(f"计算 IDF 时发生错误: {e}")
        raise

def compute_tfidf_vector(tf, idf, vocab):
    """
    计算 TF-IDF 向量
    :param tf: dict, 词频
    :param idf: dict, 逆文档频率
    :param vocab: list, 词汇表
    :return: numpy array, TF-IDF 向量
    """
    try:
        tfidf_vector = np.zeros(len(vocab))
        for idx, term in enumerate(vocab):
            tfidf_vector[idx] = tf.get(term, 0) * idf.get(term, 0)
        return tfidf_vector
    except Exception as e:
        logging.error(f"计算 TF-IDF 向量时发生错误: {e}")
        raise

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    :param vec1: numpy array, 向量 1
    :param vec2: numpy array, 向量 2
    :return: float, 余弦相似度
    """
    try:
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        if magnitude1 == 0 or magnitude2 == 0:
            logging.warning("向量模为零，无法计算余弦相似度")
            return 0.0
        return dot_product / (magnitude1 * magnitude2)
    except Exception as e:
        logging.error(f"计算余弦相似度时发生错误: {e}")
        raise

def tfidf_keyword_search(documents, query):
    """
    使用 TF-IDF 方法进行关键词匹配
    :param documents: list of str, 文档列表
    :param query: str, 查询字符串
    :return: tuple, 包含排序后的结果列表和耗时（秒）
    """
    try:
        start_time = time.time()  # 开始计时

        # 对文档和查询进行分词
        tokenized_documents = [doc.get('document_text', '').lower().split() for doc in documents]
        documents_ids = [doc.get('document_id', -1) for doc in documents]
        tokenized_query = query.lower().split()

        if not tokenized_documents:
            logging.warning("文档列表为空，无法进行关键词匹配")
            return [], 0.0

        if not tokenized_query:
            logging.warning("查询字符串为空，无法进行关键词匹配")
            return [], 0.0

        # 构建词汇表
        vocab = list(set(term for doc in tokenized_documents for term in doc))

        # 计算 IDF
        idf = compute_idf(tokenized_documents)

        # 计算每个文档的 TF-IDF 向量
        document_tfidfs = []
        for doc in tokenized_documents:
            tf = compute_tf(doc)
            tfidf_vector = compute_tfidf_vector(tf, idf, vocab)
            document_tfidfs.append(tfidf_vector)

        # 计算查询的 TF-IDF 向量
        query_tf = compute_tf(tokenized_query)
        query_tfidf = compute_tfidf_vector(query_tf, idf, vocab)

        # 计算每个文档与查询的余弦相似度
        scores = []
        for doc_tfidf, idx in zip(document_tfidfs, documents_ids):
            similarity = cosine_similarity(doc_tfidf, query_tfidf)
            scores.append((idx, similarity))

        # 按相似度得分降序排序
        ranked_results = sorted(scores, key=lambda x: x[1], reverse=True)

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        logging.info(f"关键词匹配完成，共处理 {len(documents)} 篇文档，耗时 {elapsed_time:.4f} 秒")
        return ranked_results, elapsed_time
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

    results = tfidf_keyword_search(documents, query)

    # 输出结果
    for doc_index, score in results:
        print(f"Document {doc_index + 1} score: {score:.4f}")
"""
