# 从同目录的模块导入并暴露在包级别
from .QwenLLM import (
    qwen_qa,
)

__all__ = [
    'qwen_qa'
]