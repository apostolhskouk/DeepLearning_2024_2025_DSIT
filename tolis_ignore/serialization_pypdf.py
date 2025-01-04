import os
from pypdf import PdfReader

def extract_pdf_text_to_file(pdf_path, output_folder):
    """
    Extracts text from a PDF file and saves it to a .txt file in the specified output folder.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Path to the folder where the output text file should be saved.

    Returns:
        str: Path to the generated text file.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the output folder does not exist or is not a directory.
    """
    # Check if the PDF file exists
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # Check if the output folder exists and is a directory
    if not os.path.isdir(output_folder):
        raise ValueError(f"The folder {output_folder} does not exist or is not a directory.")

    # Extract text from the PDF
    reader = PdfReader(pdf_path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() + "\n"

    # Create output text file path
    output_file_name = os.path.splitext(os.path.basename(pdf_path))[0] + "_pypdf.txt"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Save the extracted text to the file
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)

    return output_file_path


if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"

    extract_pdf_text_to_file(pdf_path, output_dir)