import os
import pdfplumber


def plumber_serialize(pdf_path, output_folder, layout=False, x_tolerance=3, y_tolerance=3):

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    pdf_name = pdf_path.split("/")[-1].split(".")[0]
    output_text_path = os.path.join(output_folder, f"{pdf_name}_plumber.txt")

    try:
        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=x_tolerance, y_tolerance=y_tolerance, layout=layout)
                if page_text:
                    all_text += page_text + "\n"

        # Write the extracted text to the output file
        with open(output_text_path, "w", encoding="utf-8") as text_file:
            text_file.write(all_text.strip())

        print(f"Text extracted and saved to: {output_text_path}")
        return output_text_path

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"
    plumber_serialize(pdf_path, output_dir)