def search_keyword_from_documents(documents, keyword):
    """
    从输入的 documents 列表中检索关键词所在条目，并返回对应的 id 列表。

    :param documents: list, 包含文档的列表，每个文档是一个字典，至少包含 'document_id' 和 'document_text' 键
    :param keyword: str, 要检索的关键词
    :return: list, 包含所有匹配条目的 id，如果没有匹配则返回空列表
    """
    found_ids = []
    
    for document in documents:
        # 检查关键词是否在当前条目的内容中
        if keyword in document.get('document_text', ''):
            found_ids.append(document.get('document_id', -1))  # 默认值为 -1，表示缺失的 ID
    
    return found_ids

# 示例调用
"""
documents = [
    {"document_id": 1, "document_text": "这是第一条内容"},
    {"document_id": 2, "document_text": "这是包含示例关键词的内容"},
    {"document_id": 3, "document_text": "这是第三条内容"}
]
keyword = '示例关键词'
result = search_keyword_from_documents(documents, keyword)
print(result)
"""
