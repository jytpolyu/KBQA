import time
import logging
import numpy as np
from gensim.models import KeyedVectors, FastText
from sklearn.metrics.pairwise import cosine_similarity

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("vector_space_search.log"),
        logging.StreamHandler()
    ]
)

def _vector_space_search(documents, query, model, top_k=None):
    """
    通用的向量空间检索函数
    :param documents: list of dict, 文档列表，每个文档包含 'document_id' 和 'document_text'
    :param query: str, 查询字符串
    :param model: gensim 模型对象 (Word2Vec, GloVe, FastText)
    :param top_k: int, 控制返回的结果数量，默认为 None 表示返回所有结果
    :return: tuple, 包含排序后的结果列表和耗时（秒）
    """
    try:
        start_time = time.time()  # 开始计时

        # 提取文档文本和文档 ID
        document_texts = [doc.get('document_text', '').lower().split() for doc in documents]
        document_ids = [doc.get('document_id', -1) for doc in documents]

        if not document_texts:
            logging.warning("文档列表为空，无法进行关键词匹配")
            return [], 0.0

        if not query.strip():
            logging.warning("查询字符串为空，无法进行关键词匹配")
            return [], 0.0

        # 将查询转换为向量
        query_vector = np.mean(
            [model[word] for word in query.lower().split() if word in model], axis=0
        ).reshape(1, -1)

        # 计算每个文档的向量
        document_vectors = []
        for doc in document_texts:
            doc_vector = np.mean(
                [model[word] for word in doc if word in model], axis=0
            )
            document_vectors.append(doc_vector)

        # 转换为 NumPy 数组
        document_vectors = np.array(document_vectors)

        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, document_vectors).flatten()

        # 将文档 ID 和相似度配对
        results = list(zip(document_ids, similarities))

        # 按相似度降序排序
        ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

        # 如果指定了 top_k，则只返回前 top_k 个结果
        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        return ranked_results, elapsed_time
    except Exception as e:
        logging.error(f"向量空间检索时发生错误: {e}")
        raise

def glove_keyword_search(documents, query, model_path, top_k=None):
    """
    使用 GloVe 模型实现关键词匹配
    """
    try:
        start_time = time.time()  # 开始计时加载模型
        model = KeyedVectors.load_word2vec_format(model_path, binary=False)  # 加载 GloVe 模型
        load_time = time.time() - start_time  # 计算加载模型的时间
        logging.info(f"GloVe 模型加载完成，耗时 {load_time:.4f} 秒")

        return _vector_space_search(documents, query, model, top_k)
    except Exception as e:
        logging.error(f"GloVe 匹配时发生错误: {e}")
        raise

def fasttext_keyword_search(documents, query, model_path, top_k=None):
    """
    使用 FastText 模型实现关键词匹配
    """
    try:
        start_time = time.time()  # 开始计时加载模型
        model = FastText.load(model_path)  # 加载 FastText 模型
        load_time = time.time() - start_time  # 计算加载模型的时间
        logging.info(f"FastText 模型加载完成，耗时 {load_time:.4f} 秒")

        return _vector_space_search(documents, query, model, top_k)
    except Exception as e:
        logging.error(f"FastText 匹配时发生错误: {e}")
        raise

def word2vec_keyword_search(documents, query, model_path, top_k=None):
    """
    使用 Word2Vec 模型实现关键词匹配
    """
    try:
        start_time = time.time()  # 开始计时加载模型
        model = KeyedVectors.load(model_path)  # 加载 Word2Vec 模型
        load_time = time.time() - start_time  # 计算加载模型的时间
        logging.info(f"Word2Vec 模型加载完成，耗时 {load_time:.4f} 秒")

        return _vector_space_search(documents, query, model, top_k)
    except Exception as e:
        logging.error(f"Word2Vec 匹配时发生错误: {e}")
        raise

# 示例调用
"""
if __name__ == "__main__":
    documents = [
        {"document_id": 1, "document_text": "This is the first document."},
        {"document_id": 2, "document_text": "This document contains example keywords."},
        {"document_id": 3, "document_text": "This is the third document."}
    ]
    query = "example keywords"
    model_path = "path/to/word2vec.model"  # 替换为实际模型路径

    results, elapsed_time = glove_keyword_search(documents, query, model_path, top_k=2)

    # 输出结果
    for doc_id, score in results:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")
    print(f"耗时：{elapsed_time:.4f} 秒")
"""