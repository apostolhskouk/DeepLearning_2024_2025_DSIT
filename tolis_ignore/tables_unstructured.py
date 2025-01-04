import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition

def extract_tables_method_1(pdf_path, output_folder):
    """
    Extract tables from a PDF file using `partition_pdf`.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Path to the folder where output files will be saved.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract tables using partition_pdf
    elements = partition_pdf(
        filename=pdf_path,
        infer_table_structure=True,
        strategy='hi_res',
    )

    tables = [el for el in elements if el.category == "Table"]

    # Save each table to a separate file
    for i, table in enumerate(tables):
        table_path = os.path.join(output_folder, f"table_method_1_{i+1}.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table.text)
        print(f"Table {i+1} saved to {table_path}")

def extract_tables_method_2(pdf_path, output_folder):
    """
    Extract tables from a PDF file using `partition`.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Path to the folder where output files will be saved.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Extract tables using partition
    elements = partition(
        filename=pdf_path,
        strategy='hi_res',
    )

    tables = [el for el in elements if el.category == "Table"]

    # Save each table to a separate file
    for i, table in enumerate(tables):
        table_path = os.path.join(output_folder, f"table_method_2_{i+1}.txt")
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(table.text)
        print(f"Table {i+1} saved to {table_path}")

# Example usage
# extract_tables_method_1("example-docs/pdf/layout-parser-paper.pdf", "output")
# extract_tables_method_2("example-docs/pdf/layout-parser-paper.pdf", "output")
if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"
    
    extract_tables_method_2(pdf_path, output_dir)