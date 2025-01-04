import os
import pymupdf  # PyMuPDF

def extract_pdf_to_text(pdf_path, output_folder):
    """
    Extracts text from a PDF and saves it into a text file in the specified output folder.

    Parameters:
        pdf_path (str): Path to the PDF file to be processed.
        output_folder (str): Path to the folder where the text file will be saved.

    Returns:
        str: Path to the generated text file.

    Raises:
        FileNotFoundError: If the PDF file or output folder does not exist.
        ValueError: If the provided paths are invalid.
    """
    # Validate input paths
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not os.path.isdir(output_folder):
        raise FileNotFoundError(f"Output folder not found: {output_folder}")

    # Extract file name without extension for naming the output file
    pdf_name = pdf_path.split("/")[-1].split(".")[0]
    output_file = os.path.join(output_folder, f"{pdf_name}_pymupdf.txt")

    # Open the PDF and extract text
    with pymupdf.open(pdf_path) as doc:
        with open(output_file, "w", encoding="utf-8") as text_file:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                text_file.write(f"\n--- Page {page_num} ---\n\n")
                text_file.write(text)

if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"
    
    extract_pdf_to_text(pdf_path, output_dir)