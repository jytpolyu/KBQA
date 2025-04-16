import faiss
import numpy as np
import time
import logging
from sentence_transformers import SentenceTransformer

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dpr_dense_faiss_search.log"),
        logging.StreamHandler()
    ]
)

def dense_faiss_search(index_file_path, question, model_name, top_k=1):
    """
    使用 FAISS 索引对传入的问题进行密集检索
    :param index_file_path: str, FAISS 索引文件路径
    :param question: str, 查询问题
    :param model_name: str, 用于生成嵌入的预训练模型名称
    :param top_k: int, 返回的文档数量
    :return: list, 包含按相似度排序的文档 ID 列表
    """
    try:
        start_time = time.time()  # 开始计时

        # 加载 FAISS 索引
        logging.info(f"加载 FAISS 索引文件：{index_file_path}")
        index = faiss.read_index(index_file_path)

        # 加载预训练的 SentenceTransformer 模型
        logging.info(f"加载预训练模型：{model_name}")
        model = SentenceTransformer(model_name)

        # 对查询问题生成嵌入
        logging.info("生成查询问题的嵌入向量...")
        query_embedding = model.encode(question, convert_to_tensor=False).astype(np.float32)

        # 使用 FAISS 索引进行检索
        logging.info("使用 FAISS 索引进行检索...")
        distances, indices = index.search(np.array([query_embedding]), top_k)

        # 将结果配对
        results = [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

        # 按相似度降序排序
        ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

        # 提取文档 ID 列表
        ranked_ids = [doc_id for doc_id, _ in ranked_results]

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        logging.info(f"DPR 检索完成，共返回 {len(results)} 个结果，耗时 {elapsed_time:.4f} 秒")
        return ranked_ids
    except Exception as e:
        logging.error(f"DPR 检索时发生错误: {e}")
        raise

"""
# 示例调用
if __name__ == "__main__":
    index_file_path = "Models/faiss_index.index"
    question = "What is the Roman Empire?"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    top_k = 5

    results = dense_faiss_search(index_file_path, question, model_name, top_k=top_k)
    print("检索结果：", results)
"""