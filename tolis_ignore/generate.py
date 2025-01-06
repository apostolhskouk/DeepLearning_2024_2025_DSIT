from PDFSerlalizer import DocumentHandler
import os
#iterate over all pdfs in the directory

input_dir = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs"
output_dir_base = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_tables"

for pdf in os.listdir(input_dir):
    pdf_path = os.path.join(input_dir, pdf)
    output_dir = os.path.join(output_dir_base, pdf.split(".")[0])
    os.makedirs(output_dir, exist_ok=True)
    doc = DocumentHandler()
    doc.extract_images(pdf_path, output_dir,verbose=True,export_pages=False,export_figures=False,export_tables=True)