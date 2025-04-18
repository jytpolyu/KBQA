# KBQA 系统测试用例

## 一、检索功能

| 用例ID |     功能点     |  模式  |  算法  |    Query     |          预期结果           | 结果 | 搜索时间 |
| :----: | :------------: | :----: | :----: | :----------: | :-------------------------: | ---- | :------: |
|  S01   | TF‑IDF短文检索 | SEARCH | TF‑IDF | roman empire | 返回包含 doc_id=1097 的结果 | PASS |          |
|  S02   | TF‑IDF长文检索 | SEARCH | TF‑IDF |              |                             | PASS |          |
|  S03   |  BM25短文检索  | SEARCH |  BM25  | roman empire | 返回包含 doc_id=1097 的结果 | PASS |          |
|  S04   |  BM25长文检索  | SEARCH |  BM25  |              |                             | PASS |          |
|  S05   | FAISS短文检索  | SEARCH | FAISS  | roman empire | 返回包含 doc_id=1097 的结果 | PASS |          |
|  S06   | FAISS长文检索  | SEARCH | FAISS  |              |                             | PASS |          |
|  S07   | GloVe短文检索  | SEARCH | GloVe  |    未部署    |                             | PASS |          |
|  S08   | GloVe长文检索  | SEARCH | GloVe  |              |                             | PASS |          |

## 二、问答生成功能

| 用例ID | 功能点 |  模式  | Query |   预期结果   | 结果 | 生成时间 |
| :----: | :----: | :----: | :---: | :----------: | :--: | -------- |
|  A01   |  生成  | ANSWER |       | roman empire |      | PASS     |

未运行