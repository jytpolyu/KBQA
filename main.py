import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import KnowledgeBase
import streamlit as st

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
def UI_Start():
    st.title("文档检索与问题回答系统")

    # 系统切换按钮
    system_mode = st.radio("选择系统模式：", ["文档检索系统", "问题回答生成系统"])

    if system_mode == "文档检索系统":
        st.header("文档检索系统")

        # 搜索算法选择
        algorithm = st.selectbox("选择搜索算法：", ["TF-IDF", "BM25", "其他算法"])

        # 输入框
        query = st.text_input("请输入查询内容：", "")

        # 检索按钮
        if st.button("检索"):
            if query.strip():
                top_k = 5  # 默认返回前 5 条结果
                start_time = time.time()  # 开始计时

                # 根据选择的算法执行不同的搜索
                if algorithm == "TF-IDF":
                    results = KnowledgeBase.search_keyword_from_documents(engine.documents, query)
                elif algorithm == "BM25":
                    results = KnowledgeBase.search_with_bm25(engine.documents, query)
                else:
                    st.write("暂不支持该算法。")
                    return

                end_time = time.time()  # 结束计时
                elapsed_time = end_time - start_time

                # 显示搜索结果
                if results:
                    st.write(f"找到 {len(results)} 条结果：")
                    output_content = ""
                    for result in results:
                        output_content += f"**ID**: {result['document_id']}\n"
                        output_content += f"**Content**: {result['document_text']}\n"
                        output_content += "---\n"
                    st.text_area("输出内容：", output_content, height=300)
                else:
                    st.write("未找到相关结果。")

                # 显示搜索时间
                st.write(f"搜索时间：{elapsed_time:.4f} 秒")
            else:
                st.write("请输入有效的查询内容。")

    elif system_mode == "问题回答生成系统":
        st.header("问题回答生成系统")

        # 输入框
        question = st.text_input("请输入您的问题：", "")

        # 生成回答按钮
        """
        if st.button("生成回答"):
            if question.strip():
                start_time = time.time()  # 开始计时

                # 调用问题回答生成函数
                answer = KnowledgeBase.generate_answer(question)

                end_time = time.time()  # 结束计时
                elapsed_time = end_time - start_time

                # 显示生成的回答
                if answer:
                    st.write("生成的回答：")
                    st.text_area("回答内容：", answer, height=150)
                else:
                    st.write("未能生成回答，请尝试其他问题。")

                # 显示生成时间
                st.write(f"生成时间：{elapsed_time:.4f} 秒")
            else:
                st.write("请输入有效的问题。")
        """

if __name__ == "__main__":
    file_path = "data/documents.jsonl"  # 替换为你的文件路径
    engine = DocumentSearchEngine(file_path)
    query = 'roman empire'
    results, elpasetime = KnowledgeBase.search_keyword_from_documents(engine.documents, query)
    print(results)
    #UI_Start()