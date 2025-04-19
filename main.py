# 说明：
# 本文件是 KBQA 系统的主入口，基于 FastAPI 实现后端服务。
# 功能模块：
# 1. 文档搜索引擎类：加载文档并支持多种检索算法（TF-IDF、BM25、FAISS、GloVe）。
# 2. FastAPI 接口：提供文档检索和问答服务的 RESTful API。
# 3. 前端静态文件挂载：支持前端页面的访问。
# 4. 支持多种检索模式（SEARCH、ANSWER）和算法。
import time
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.metrics.pairwise import cosine_similarity
import KnowledgeBase
import AnswerGeneration
import EngineClass

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
engine = EngineClass.DocumentSearchEngine(file_path)

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
    
    # 首先判断模式
    if mode == "ANSWER":
        # ANSWER模式固定使用FAISS和问答模型
        doc_ids = KnowledgeBase.dense_faiss_search(engine.index_file_path, query, engine.model_name, 1)
        documents = engine._get_documents_by_id(doc_ids)
        results = AnswerGeneration.qwen_qa(documents, query, engine.api_url, engine.api_key)
    elif mode == "SEARCH":
        # SEARCH模式根据method选择不同搜索方法
        if method == "TF-IDF":
            doc_ids = KnowledgeBase.tfidf_keyword_search(engine.documents, query, 1)
            results = engine._get_documents_by_id(doc_ids)
        elif method == "BM25":
            doc_ids = KnowledgeBase.bm25_keyword_search(engine.documents, query)
            results = engine._get_documents_by_id(doc_ids)
        elif method == "FAISS":
            doc_ids = KnowledgeBase.dense_faiss_search(engine.index_file_path, query, engine.model_name, 1)
            results = engine._get_documents_by_id(doc_ids)
        elif method == "GloVe":
            doc_ids = KnowledgeBase.glove_keyword_search(engine.documents, query, engine.glove_path, top_k=1)
            results = engine._get_documents_by_id(doc_ids)
        else:
            return {"error": f"暂不支持的算法：{method}"}
    else:
        return {"error": f"暂不支持的模式：{mode}"}
    
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
