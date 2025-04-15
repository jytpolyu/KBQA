import json
import time
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import KnowledgeBase

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

    def _load_documents(self):
        start_time = time.time()
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line.strip())
                self.documents.append(doc)
        end_time = time.time()
        print(f"文档加载完成，共加载 {len(self.documents)} 条记录。")
        print(f"加载时间：{end_time - start_time:.4f} 秒")

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
            results = [
                {"document_id": 1234, "document_text": "文档1"},
                {"document_id": 4567, "document_text": "文档2"},
            ]
            # results = KnowledgeBase.search_keyword_from_documents(engine.documents, query)
        elif mode == "ANSWER":
            results = "这是我的答案"
            # results = KnowledgeBase.answer_from_documents(engine.documents, query)
    elif method == "BM25":
        results = "bbb"
        # results = KnowledgeBase.search_with_bm25(engine.documents, query)
    else:
        return {"error": f"暂不支持的算法：{method}"}
    end_time = time.time()

    return {
        "results": results,
        "elapsed_time": round(end_time - start_time, 4)
    }


if __name__ == "__main__":
    file_path = "data/documents.jsonl"  # 替换为你的文件路径
    # engine = DocumentSearchEngine(file_path)
    # query = 'roman empire'
    # results, elpasetime = KnowledgeBase.bm25_keyword_search(engine.documents, query)
    # print(results,elpasetime)

    # results, elpasetime = KnowledgeBase.tfidf_keyword_search(engine.documents, query)
    # print(results,elpasetime)

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)