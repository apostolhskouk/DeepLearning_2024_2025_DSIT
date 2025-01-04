from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import os 

def marker_serialize(pdf_path, output_folder,output_format="markdown") :
    #check if directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    config = {
        "output_format": output_format,
    }
    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer()
    )
    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)
    # extract the name file of the pdf path
    pdf_name = pdf_path.split("/")[-1].split(".")[0]
    if output_format == "markdown":
        with open(os.path.join(output_folder, f"{pdf_name}_marker.md"), "w") as f:
            f.write(text)
    elif output_format == "html":
        with open(os.path.join(output_folder, f"{pdf_name}_marker.html"), "w") as f:
            f.write(text)
    elif output_format == "json":
        with open(os.path.join(output_folder, f"{pdf_name}_marker.json"), "w") as f:
            f.write(rendered.to_json())



if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output"
    marker_serialize(pdf_path, output_dir, "markdown")
