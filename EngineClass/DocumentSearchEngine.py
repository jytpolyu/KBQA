# 说明：
# 本文件实现了文档搜索引擎的核心功能，并提供与 Qwen 模型相关的配置。
# 功能模块：
# 1. 文档加载：从 JSONL 文件加载文档数据，并支持去除 HTML 标签。
# 2. 文档检索：根据文档 ID 检索对应的文档内容。
# 3. 模型配置：提供与 Qwen 模型相关的参数配置（如模型名称、API URL 和密钥）。
import time
import json
import re

# -----------------------
# 文档搜索引擎类
# -----------------------
class DocumentSearchEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.documents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_documents()
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.api_url = r"https://api.siliconflow.cn/v1"
        self.api_key = "sk-lwngutjeflildxuguicsqjkzzqpnnjiyuwldtaljmpgimwyl"
        self.index_file_path = "Models/faiss_index.index"
        self.glove_path = "Models/glove.6B.300d.word2ve.txt"
        self.fasttext_path = "Models/wiki-news-300d-1M-subword.vec"

    def _load_documents(self):
        """
        加载 JSONL 文件中的文档，并记录加载时间
        """
        start_time = time.time()  # 开始计时
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line.strip())
                doc['document_text'] = re.sub(r'<[^>]+>', '', doc['document_text'])  # 去除 HTML 标签
                self.documents.append(doc)
        end_time = time.time()
        print(f"文档加载完成，共加载 {len(self.documents)} 条记录。")
        print(f"加载时间：{end_time - start_time:.4f} 秒")

    def _get_documents_by_id(self, results):
        """
        根据 results 列表中的文档 ID，从 documents 中找到对应的文本并打印
        :param documents: list of dict, 文档列表，每个文档包含 'document_id' 和 'document_text'
        :param results: list of tuple, 包含文档 ID 和分数的元组列表
        """
        docs_list = []
        for doc_id in results:
            # 查找对应的文档
            doc = next((doc for doc in self.documents if doc['document_id'] == doc_id), None)
            if doc:
                docs_list.append(doc)
            else:
                print(f"Document ID: {doc_id} 未找到对应的文档")
        
        return docs_list