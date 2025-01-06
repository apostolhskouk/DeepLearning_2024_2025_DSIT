from byaldi import RAGMultiModalModel
import os
from pathlib import Path

# Optionally, you can specify an `index_root`, which is where it'll save the index. It defaults to ".byaldi/".
model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0", index_root="./index_byaldi")

metadata = [{"filename":file_name} for file_name in os.listdir("/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_tables/")]

index_name = "cs_ai_2023_tables"

model.index(
    input_path=Path("/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_tables/"),
    index_name=index_name,
    store_collection_with_index=False,
    metadata=metadata,
    overwrite=True
)