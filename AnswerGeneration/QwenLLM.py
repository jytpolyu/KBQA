# 说明：
# 本文件用于与 Qwen API 交互，基于输入的文档和问题生成答案。
# 功能模块：
# 1. 初始化 Qwen 客户端。
# 2. 构造请求消息，包括问题和相关文档内容。
# 3. 调用 Qwen API 生成答案，并限制答案长度为 10 个单词。
# 4. 解析 API 响应并返回答案。
import logging
import time
from openai import OpenAI

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("qwen_qa.log"),
        logging.StreamHandler()
    ]
)

def qwen_qa(documents, question, api_url, api_key):
    """
    使用 Qwen 的 API 基于输入的 documents 数据进行问答
    :param documents: list, 包含文档内容的列表
    :param question: str, 用户提出的问题
    :param api_url: str, Qwen API 的 URL
    :param api_key: str, 用于认证的 API 密钥
    :return: str, Qwen API 返回的答案
    """
    try:
        start_time = time.time()  # 开始计时

        logging.info("初始化 Qwen 客户端...")
        client = OpenAI(api_key=api_key, base_url=api_url)

        # 构造请求消息
        logging.info("构造请求消息...")
        messages = [{'role': 'user', 'content': question}]
        if documents:
            max_length = 2000 // len(documents)
            context = "\n".join([doc.get('document_text', '')[:max_length] for doc in documents])
            messages.insert(0, {'role': 'system', 'content': f"The following are relevant document contents:\n{context}. Please answer in English with a maximum of ten words."})

        # 调用 Qwen API
        logging.info("调用 Qwen API...")
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=messages
        )

        # 检查响应结构
        #logging.info(f"Qwen API 响应内容: {response}")

        # 解析响应
        logging.info("解析 Qwen API 响应...")
        answer = ""
        try:
            # 遍历 response 中的 choices
            if response:
                message_content = response.choices[0].message.content
                answer = " ".join(message_content.split()[:10])
            else:
                logging.info("Qwen 返回的答案为空，请检查网络连接")
                    
        except Exception as e:
            logging.error(f"解析 Qwen API 响应时发生错误: {e}")
            return "未能生成答案"

        if answer:
            end_time = time.time()  # 结束计时
            elapsed_time = end_time - start_time
            logging.info(f"Qwen QA 完成，耗时 {elapsed_time:.4f} 秒")
            return answer
        else:
            logging.warning("Qwen API 未返回有效答案")
            return "未能生成答案"

    except Exception as e:
        logging.error(f"Qwen QA 过程中发生错误: {e}")
        raise

"""
# 示例调用
if __name__ == "__main__":
    documents = [
        {"document_id": 1, "document_text": "The Roman Empire was one of the largest empires in history."},
        {"document_id": 2, "document_text": "It was founded in 27 BC and lasted until 476 AD in the West."}
    ]
    question = "What is the Roman Empire?"
    api_url = "https://api.siliconflow.cn/v1"
    api_key = "sk-lwngutjeflildxuguicsqjkzzqpnnjiyuwldtaljmpgimwyl"

    try:
        answer = qwen_qa(documents, question, api_url, api_key)
        print(f"AI 答案: {answer}")
    except Exception as e:
        logging.error(f"主程序运行时发生错误: {e}")
"""