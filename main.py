import json
import time
import uvicorn
import os
import re
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.metrics.pairwise import cosine_similarity
import KnowledgeBase
import AnswerGeneration

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

# -----------------------
# 初始化 FastAPI 应用
# -----------------------
app = FastAPI()

# 获取绝对路径
dist_path = os.path.join(os.path.dirname(__file__), "client", "dist")

# 挂载前端静态文件目录
app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

# 根路径返回 index.html
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(dist_path, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# 初始化文档引擎
# -----------------------
file_path = "data/documents.jsonl"
engine = DocumentSearchEngine(file_path)

# -----------------------
# FastAPI 接口定义
# -----------------------
@app.post("/api/search")
async def search(request: Request):
    params = await request.json()
    print(f"[接口调用] 接收到的参数：{params}")  # 打印所有参数
    query = params.get("query", "").strip()
    method = params.get("method", "TF-IDF").upper()
    mode = params.get("mode", "default").upper()
    print(f"[接口调用] 查询：{query} | 方法：{method} | 模式：{mode}")

    if not query:
        return {"error": "查询内容不能为空"}

    start_time = time.time()
    results = ''
    if method == "TF-IDF":
        if mode == "SEARCH":
            """
            results = [
                {"document_id": 1234, "document_text": "文档1"},
                {"document_id": 4567, "document_text": "文档2"},
            ]
            """
            results = KnowledgeBase.tfidf_keyword_search(engine.documents, query, 1)
            results = engine._get_documents_by_id(results)

        elif mode == "ANSWER":
            #results = "这是我的答案"
            results = KnowledgeBase.dense_faiss_search(engine.index_file_path,query,engine.model_name,1)

            document = engine._get_documents_by_id(results)

            results = AnswerGeneration.qwen_qa(document, query)
    elif method == "BM25":
        #results = "bbb"
        results = KnowledgeBase.bm25_keyword_search(engine.documents, query)
        results = engine._get_documents_by_id(results)
    elif method == "FAISS":
        #results = "bbb"
        results = KnowledgeBase.dense_faiss_search(engine.index_file_path,query,engine.model_name,1)
        results = engine._get_documents_by_id(results)
    elif method == "GloVe":
        #results = "bbb"
        results = KnowledgeBase.glove_keyword_search(engine.documents, query, engine.glove_path, top_k=1)
        results = engine._get_documents_by_id(results)
    else:
        return {"error": f"暂不支持的算法：{method}"}
    end_time = time.time()

    return {
        "results": results,
        "elapsed_time": round(end_time - start_time, 4)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

    """
    query = 'roman empire'  # 测试查询,正式使用可以删除

    #使用Qwen模型进行问答的流程，首先使用FAISS搜索确定相关文档，然后将相关文档作为上下文输入到Qwen模型中进行问答
    results, elpasetime = KnowledgeBase.dense_faiss_search(index_file_path,query,engine.model_name,1)

    document = engine._get_documents_by_id(results)

    results, elpasetime = AnswerGeneration.qwen_qa(document, query, engine.api_url, engine.api_key)

    #使用FAISS进行相似度搜索
    results, elpasetime = KnowledgeBase.dense_faiss_search(index_file_path,query,engine.model_name,1)

    #使用BM25进行相似度搜索
    results, elpasetime = KnowledgeBase.bm25_keyword_search(engine.documents, query, top_k=1)

    #使用tfidf进行相似度搜索
    results, elpasetime = KnowledgeBase.tfidf_keyword_search(engine.documents, query, top_k=1)

    #使用GloVe进行相似度搜索
    results, elpasetime = KnowledgeBase.glove_keyword_search(engine.documents, query, glove_path, top_k=1)

    """
