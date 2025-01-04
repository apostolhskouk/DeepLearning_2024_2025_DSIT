from markitdown import MarkItDown
import os 

def markitdown_serialize(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    md = MarkItDown()
    result = md.convert(pdf_path)
    pdf_name = pdf_path.split("/")[-1].split(".")[0]
    with open(os.path.join(output_folder, f"{pdf_name}_markitdown.md"), "w") as f:
        f.write(result.text_content)
    
    
if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"
    markitdown_serialize(pdf_path, output_dir)