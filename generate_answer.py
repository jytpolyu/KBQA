# 说明：
# 本文件主要用于实现预测文件prediction.jsonl的生成。
# 功能模块：
# 1. 文档搜索引擎类：加载文档并支持根据文档 ID 检索内容。
# 2. 问题处理：从测试文件中提取问题。
# 3. 答案生成：通过调用本地模型或 API 生成答案。
# 4. 结果保存：将生成的答案和相关文档 ID 保存为 JSONL 文件。

import json
import time
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import KnowledgeBase
import requests
import AnswerGeneration

# -----------------------
# 文档搜索引擎类
# -----------------------
class DocumentSearchEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.documents = {}  # 改为字典类型，键为 document_id，值为文档内容
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
                # 使用 document_id 作为键，文档内容作为值
                self.documents[doc['document_id']] = doc
        end_time = time.time()
        print(f"文档加载完成，共加载 {len(self.documents)} 条记录。")
        print(f"加载时间：{end_time - start_time:.4f} 秒")

    def _get_documents_by_id(self, results):
        """
        根据 results 列表中的文档 ID，从 documents 中找到对应的文本并返回
        :param results: list of str, 包含文档 ID 的列表
        :return: list of dict, 对应的文档内容列表
        """
        docs_list = []
        for doc_id in results:
            # 从字典中快速查找文档
            doc = self.documents.get(doc_id, None)
            if doc:
                docs_list.append(doc)
        return docs_list

def get_test_questions(test_file_path):
    question_list = []

    # 读取 JSONL 文件并提取问题
    with open(test_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            question = data.get("question", None)
            if question:  # 如果问题存在，则添加到列表
                question_list.append(question)

    print(f"从 {test_file_path} 中加载了 {len(question_list)} 个问题")
    #print("示例问题:", question_list[:5])  # 打印前 5 个问题
    return question_list

def get_answer_by_api(api_url, api_key, question_list, file_path):
    """
    使用千问的 API 和预训练模型生成回答。
    :param api_url: str, 千问 API 的 URL。
    :param api_key: str, 千问 API 的密钥。
    :param question_list: list, 包含问题的列表。
    :param file_path: str, 文档数据文件路径，用于检索相关上下文。
    :return: tuple, 包含生成的答案列表 (answer_list) 和相关文档 ID 列表 (id_list)。
    """
    answer_list = []
    id_list = []

    # 初始化文档搜索引擎
    engine = DocumentSearchEngine(file_path)

    for question in question_list:
        # 使用密集检索获取相关文档 ID
        relevant_ids = KnowledgeBase.dense_faiss_search(engine.index_file_path, question, engine.model_name, 5)
        id_list.append(relevant_ids)

        # 获取相关文档内容
        relevant_texts = engine._get_documents_by_id(relevant_ids)

        # 将相关文档内容拼接为上下文
        max_length = 2000 // len(relevant_texts)
        context = "\n".join([doc.get('document_text', '')[:max_length] for doc in relevant_texts])

        # 构造 API 请求的输入数据
        payload = {
            "question": question,
            "context": context
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 调用 API 获取回答
        response = requests.post(api_url, headers=headers, json=payload)

        if response.status_code == 200:
            answer = response.json().get("answer", "").strip()
        else:
            print(f"API 调用失败，状态码: {response.status_code}, 响应: {response.text}")
            answer = "无法生成回答"

        answer_list.append(answer)

    return answer_list, id_list

def get_answer_by_RAG(model_name, model_path, tokenizer_path, question_list, file_path):
    """
    使用检索增强生成（RAG）方法生成答案。
    :param model_name: str, 使用的基础模型。
    :param model_path: str, 微调后的 LoRA 模型路径。
    :param tokenizer_path: str, 分词器路径。
    :param question_list: list, 包含问题的列表。
    :param file_path: str, 文档数据文件路径，用于检索相关上下文。
    :return: tuple, 包含生成的答案列表 (answer_list) 和相关文档 ID 列表 (id_list)。
    """
    answer_list = []
    id_list = []

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 配置 4-bit 量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # 设置为 FP16
    )

    # 加载 LoRA 配置
    peft_config = PeftConfig.from_pretrained(model_path)

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # 加载 LoRA 微调模型
    model = PeftModel.from_pretrained(base_model, model_path)

    # 确保模型在 GPU 上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for question in question_list:
        # 使用密集检索获取相关文档 ID
        relevant_ids = KnowledgeBase.dense_faiss_search(engine.index_file_path, question, engine.model_name, 5)
        id_list.append(relevant_ids)

        # 获取相关文档内容
        relevant_texts = engine._get_documents_by_id(relevant_ids)

        # 将相关文档内容拼接为上下文
        max_length = 2000 // len(relevant_texts)
        context = "\n".join([doc.get('document_text', '')[:max_length] for doc in relevant_texts])

        # 构造输入文本
        input_text = f"question: {question},context: {context}.answer:"

        # 编码输入
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2200,
            padding="max_length"
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 生成回答
        outputs = model.generate(
            **inputs,
            num_beams=3,  # 使用 beam search 提升生成质量
            early_stopping=True,
            repetition_penalty=1.2,      # 避免重复
            no_repeat_ngram_size=2,      # 避免生成重复短语
            temperature=0.1             # 控制随机性
        )

        # 解码生成的回答
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去除与输入文本重复的部分
        if input_text in answer:
            answer = answer.replace(input_text, "").strip()
        answer_list.append(answer)

    return answer_list, id_list

def write_predictions_to_jsonl(prediction_file_path, question_list, answer_list, id_list):
    """
    将生成的答案和相关文档 ID 写入 JSONL 文件
    :param prediction_file_path: str, 输出预测结果的文件路径
    :param question_list: list, 问题列表
    :param answer_list: list, 模型生成的答案列表
    :param id_list: list, 每个问题对应的相关文档 ID 列表
    """

    # 写入预测结果
    with open(prediction_file_path, 'w', encoding='utf-8') as prediction_file:
        for question,answer,idx in zip(question_list,answer_list,id_list):
            prediction = {
                "question": question,
                "answer": answer,
                "document_id": idx
            }

            prediction_file.write(json.dumps(prediction, ensure_ascii=False) + '\n')

    print(f"预测结果已写入 {prediction_file_path}")

def load_ids_from_txt(id_txt_path):
    """
    从 TXT 文件中读取文档 ID 列表，并将每个 ID 转换为整型
    :param id_txt_path: str, 文档 ID 列表的文件路径
    :return: list, 每行对应的文档 ID 列表（整型）
    """
    id_list = []
    with open(id_txt_path, 'r', encoding='utf-8') as id_file:
        for line in id_file:
            # 将每行的文档 ID 分割成列表，去除换行符，并转换为整型
            ids = [int(doc_id) for doc_id in line.strip().split(',') if doc_id.strip().isdigit()] if line.strip() else []
            id_list.append(ids)
    return id_list

if __name__ == "__main__":
    model_path = "Models/qwen_model_lora"
    tokenizer_path = "Models/qwen_tokenizer_lora"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    test_file_path = "data/test.jsonl"
    prediction_file_path = "data/prediction.jsonl"
    file_path = "data/documents.jsonl"
    question_txt_path = "data/questions.txt"
    id_txt_path = "data/ids.txt"

    # 初始化文档搜索引擎
    engine = DocumentSearchEngine(file_path)
    question_list = get_test_questions(test_file_path)
    id_list = []
    answer_list = []

    """"""
    for question in question_list:
        ids = KnowledgeBase.dense_faiss_search(engine.index_file_path,question,engine.model_name,5)
        id_list.append([str(doc_id) for doc_id in ids])
    with open(question_txt_path, 'w', encoding='utf-8') as question_file:
        for question in question_list:
            question_file.write(question + '\n')
    print(f"问题列表已保存到 {question_txt_path}")
    # 保存文档 ID 列表
    with open(id_txt_path, 'w', encoding='utf-8') as id_file:
        for ids in id_list:
            id_file.write(','.join(ids) + '\n' if ids else '\n')
    print(f"文档 ID 列表已保存到 {id_txt_path}")
    
    id_list = load_ids_from_txt(id_txt_path)
    for question,id in zip(question_list,id_list):
       documents = engine._get_documents_by_id(id)
       answer = AnswerGeneration.qwen_qa(documents,question,engine.api_url,engine.api_key)
       answer_list.append(answer)
       
    write_predictions_to_jsonl(prediction_file_path, question_list, answer_list, id_list)