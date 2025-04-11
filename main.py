import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import KnowledgeBase
import streamlit as st  # 引入 Streamlit

# filepath: d:\KBQA\document_search_with_timer.py
class DocumentSearchEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.documents = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_documents()

    def _load_documents(self):
        """加载 JSONL 文件中的文档，并记录加载时间"""
        start_time = time.time()  # 开始计时
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                doc = json.loads(line.strip())
                self.documents.append(doc)
        end_time = time.time()  # 结束计时
        elapsed_time = end_time - start_time
        print(f"文档加载完成，共加载 {len(self.documents)} 条记录。")
        print(f"加载时间：{elapsed_time:.4f} 秒")

# Streamlit UI
def main():
    st.title("文档检索系统")
    file_path = "data/documents.jsonl"  # 替换为你的文件路径
    engine = DocumentSearchEngine(file_path)

    # 构建 TF-IDF 矩阵
    #engine._build_tfidf_matrix()

    # 输入框
    query = st.text_input("请输入查询内容：", "")

    # 检索按钮
    if st.button("检索"):
        if query.strip():
            top_k = 5  # 默认返回前 5 条结果
            results = KnowledgeBase.search_keyword_from_documents(engine.documents, query)
            if results:
                st.write(f"找到 {len(results)} 条结果：")
                for result in results:
                    st.write(f"**ID**: {result['document_id']}")
                    st.write(f"**Content**: {result['document_text']}")
                    st.write("---")
            else:
                st.write("未找到相关结果。")
        else:
            st.write("请输入有效的查询内容。")


if __name__ == "__main__":
    main()