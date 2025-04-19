# KBQA 系统

## 项目简介
KBQA（Knowledge-Based Question Answering）系统是一个基于知识库的问答系统，支持多种检索算法（如 TF-IDF、BM25、FAISS、GloVe）和问答模型（如 Qwen 模型）。系统通过 FastAPI 提供 RESTful API 服务，并支持前端页面访问。
项目Google Drive 链接：https://drive.google.com/file/d/1U89ROtJTTnRFH2BoYvD8ubOxYYXcGPLf/view?usp=sharing
解压后按照“环境配置”和“启动方法”步骤执行即可。

## 项目人员
JIANG Yutong				 
ZHAN Tanchen				 
LIU Xingyu					 

## 项目结构
├── AnswerGeneration/ # 问答生成模块  
│ ├── __init__.py  
│ ├── QwenLLM.py # 与 Qwen 模型交互的核心逻辑  
├── client/ # 前端代码  
│ ├── src/ # 前端源码  
│ ├── public/ # 静态资源  
│ ├── index.html # 前端入口文件  
│ ├── vite.config.js # Vite 配置文件  
│ ├── package.json # 前端依赖配置  
├── EngineClass/ # 文档搜索引擎模块  
│ ├── __init__.py  
│ ├── DocumentSearchEngine.py # 文档加载与检索核心逻辑  
├── KnowledgeBase/ # 检索算法模块  
│ ├── __init__.py  
│ ├── BM25_method.py # 基于 BM25 的检索算法  
│ ├── DPR_method.py # 基于密集检索（DPR）的算法  
│ ├── TF_IDF_Keyword_Match.py # 基于 TF-IDF 的检索算法  
│ ├── Vector_Space_Model.py # 基于向量空间模型的检索算法  
├── Models/ # 模型文件  
│ ├── faiss_index.index # FAISS 索引文件  
│ ├── glove.6B.300d.word2ve.txt # GloVe 词向量文件  
│ ├── qwen_model_lora/ # Qwen 模型文件  
├── data/ # 数据文件  
│ ├── documents.jsonl # 文档数据  
│ ├── train.jsonl # 训练数据  
│ ├── val.jsonl # 验证数据  
├── main.py # 系统主入口  
├── train_model.py # 模型训练脚本  
├── generate_answer.py # 生成预测答案脚本  
├── requirements.txt # Python 依赖包  
└── README.md # 项目说明文件

## 功能介绍
KBQA 系统提供以下主要功能：

1. **文档检索**：
   - 支持多种检索算法，包括 TF-IDF、BM25、FAISS 和基于向量空间模型的检索。
   - 提供高效的文档搜索功能，快速定位与查询相关的文档。

2. **问答生成**：
   - 集成 Qwen 模型，通过上下文生成高质量的答案。
   - 支持基于知识库的问答，结合检索结果生成精准回答。

3. **多模式支持**：
   - **SEARCH 模式**：仅返回与查询相关的文档。
   - **ANSWER 模式**：结合检索结果生成答案。

4. **前后端分离架构**：
   - 前端基于现代化框架构建，提供用户友好的界面。
   - 后端基于 FastAPI 实现，提供高性能的 RESTful API 服务。

5. **可扩展性**：
   - 支持自定义检索算法和问答模型的集成。
   - 可通过配置文件调整模型路径和检索参数。

6. **日志记录**：
   - 系统运行过程中会记录详细的日志，便于调试和性能优化。

## 环境配置

在运行项目之前，请确保已配置以下环境：

### 1. Python 环境
- **版本要求**：Python 3.8 或更高版本
- **依赖库**：
  使用以下命令安装所需的 Python 依赖：
  pip install -r requirements.txt

### 2. Node.js 环境
- **版本要求**：Node.js 安装v18.x.x 版本(可以使用fnm管理node版本) 

### 3. 预训练模型
- **模型路径**：如果您是通过Google Drive链接下载的压缩包，直接解压即可。如果不是，请在main.py相同路径下创建Models和data文件夹用以存储预训练模型文件和训练数据集，
- **Models**：该文件夹主要存放微调后的Qwen模型、FAISS索引和GloVe模型
- **data**：该文件夹主要jsonl格式的训练数据集

## 启动方法
### 1. UI界面
    -在KBQA目录下以管理员权限启动命令框
    -确保网络连接正常
    -使用以下命令启动程序:
        python main.py
    -访问http://127.0.0.1:8000/
    -进入界面，体验各项功能

### 2. prediction.jsonl生成
    -使用以下命令生成prediction.jsonl：
    python generate_answer.py
    -生成的prediction.jsonl在data目录下即可查看
