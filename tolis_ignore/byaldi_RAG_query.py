from byaldi import RAGMultiModalModel
index_name = "cs_ai_2023_tables"

model = RAGMultiModalModel.from_index(index_path=index_name, index_root="/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_tables/index_byaldi")

query = """
                                                                                                                                                | 63.68                  | 68.15                      |             | 25.67       | 18.01       | 30.51                 | 17.86       | 28.64       | 28.32                    | 33.47                    | 30.2        | 19.77       | 24.0      |                                  |                                  |                       | 27.7        | 15.83                | 25.76      | 25.76           | 25.76           |                | $5,813                      | $1.49                      | 18%                                     | 65                                     |
"""

results = model.search(query, k=5)

for result in results:
    print(f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score} metadata {result.metadata}")