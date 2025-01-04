import os
import pdfplumber
import csv

def extract_tables_from_pdf(pdf_path, output_folder):
    """
    Extracts tables from a PDF file and saves them as CSV files in the specified output folder.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Path to the folder where the extracted tables will be saved.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                # Use customized table settings to reduce false positives
                table_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3
                }
                tables = page.extract_tables(table_settings=table_settings)

                for table_index, table in enumerate(tables, start=1):
                    if not table or all(not any(row) for row in table):
                        continue  # Skip empty or invalid tables

                    output_file = os.path.join(
                        output_folder, f"page_{page_number}_table_{table_index}.csv"
                    )

                    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerows(table)

                    print(f"Table saved: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"
    extract_tables_from_pdf(pdf_path, output_dir)