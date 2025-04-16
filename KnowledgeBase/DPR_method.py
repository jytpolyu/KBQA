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

def dense_faiss_search(index_file_path, question, model_name, top_k):
    """
    使用 FAISS 索引对传入的问题进行密集检索
    :param index_file_path: str, FAISS 索引文件路径
    :param question: str, 查询问题
    :param model_name: str, 用于生成嵌入的预训练模型名称
    :param top_k: int, 返回的文档数量
    :return: list of tuple, 包含文档索引和分数的结果
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

        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time  # 计算耗时

        logging.info(f"DPR 检索完成，共返回 {len(results)} 个结果，耗时 {elapsed_time:.4f} 秒")
        return results, elapsed_time
    except Exception as e:
        logging.error(f"DPR 检索时发生错误: {e}")
        raise

"""
# 示例调用
if __name__ == "__main__":
    index_file_path = "data/faiss_index.bin"
    question = "What is the Roman Empire?"
    top_k = 5

    try:
        results, elapsed_time = dense_faiss_search(index_file_path, question, top_k=top_k)
        logging.info("检索结果：")
        for idx, score in results:
            logging.info(f"文档索引: {idx}, 距离: {score:.4f}")
    except Exception as e:
        logging.error(f"主程序运行时发生错误: {e}")
"""