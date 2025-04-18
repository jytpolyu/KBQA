# KBQA system test cases

## 1. Search function

| Use Case ID |    Function    | Model  | Algorithm |                           Query                           |           Expected Results            | Results | Processing Time(s) |
| :---------: | :------------: | :----: | :-------: | :-------------------------------------------------------: | :-----------------------------------: | ------- | :----------------: |
|     S01     | TF‑IDF短文检索 | SEARCH |  TF‑IDF   |                       roman empire                        | Return results containing doc_id=1097 | PASS    |     26.5293(s)     |
|     S02     | TF‑IDF长文检索 | SEARCH |  TF‑IDF   | what is the primary function of the endoplasmic reticulum | Return results containing doc_id=4974 | PASS    |     28.0923(s)     |
|     S03     |  BM25短文检索  | SEARCH |   BM25    |                       roman empire                        | Return results containing doc_id=1097 | PASS    |     18.9302(s)     |
|     S04     |  BM25长文检索  | SEARCH |   BM25    | what is the primary function of the endoplasmic reticulum | Return results containing doc_id=4974 | F(3809) |     19.3695(s)     |
|     S05     | FAISS短文检索  | SEARCH |   FAISS   |                       roman empire                        | Return results containing doc_id=1097 | PASS    |     11.0496(s)     |
|     S06     | FAISS长文检索  | SEARCH |   FAISS   | what is the primary function of the endoplasmic reticulum | Return results containing doc_id=4974 | PASS    |     4.5733(s)      |
|     S07     | GloVe短文检索  | SEARCH |   GloVe   |                          未部署                           |                                       | PASS    |                    |
|     S08     | GloVe长文检索  | SEARCH |   GloVe   |                                                           |                                       | PASS    |                    |

## 2. Question and answer generation function

| Use Case ID | Function | Model  | Query | Expected Results | Results | Processing Time |
| :---------: | :------: | :----: | :---: | :--------------: | :-----: | --------------- |
|     A01     |          | ANSWER |       |   roman empire   |         | PASS            |







 {"question": "what is the primary function of the endoplasmic reticulum", "answer": "transport of synthesized proteins", "document_id": 4974}