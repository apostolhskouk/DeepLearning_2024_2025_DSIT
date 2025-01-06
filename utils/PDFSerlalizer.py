import os
import time
import json
from pathlib import Path
import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from bs4 import BeautifulSoup
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

class DocumentHandler:
    """
    A class for handling document processing tasks, including converting PDFs to various formats
    and exporting tables from PDFs to structured formats.
    """

    def __init__(self):
        """
        Initializes the DocumentHandler instance with a DocumentConverter.
        """

    def docling_serialize(self, pdf_path, output_folder, mode=None, output_format="markdown", verbose=False):
        """
        Converts a PDF to various output formats using Docling.

        Args:
            pdf_path (str): The path to the input PDF file.
            output_folder (str): The directory where the output files will be saved.
            mode (str, optional): The table extraction mode. Options are "accurate" or None (default).
            output_format (str, optional): The desired output format. Options include:
                - "markdown" (default)
                - "json"
                - "html"
                - "indexed_text"
            verbose (bool, optional): If True, prints timing and progress information.
        
        Raises:
            ValueError: If the specified output format is unsupported.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        if mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        start_time = time.time()
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        result = converter.convert(pdf_path)
        if verbose:
            print(f"Conversion took {time.time() - start_time:.2f} seconds")

        pdf_name = Path(pdf_path).stem

        if output_format == "markdown":
            output_file = os.path.join(output_folder, f"{pdf_name}_docling.md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())
        elif output_format == "json":
            output_file = os.path.join(output_folder, f"{pdf_name}_docling.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.document.export_to_dict(), f, indent=4)
        elif output_format == "html":
            output_file = os.path.join(output_folder, f"{pdf_name}_docling.html")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.document.export_to_html())
        elif output_format == "indexed_text":
            output_file = os.path.join(output_folder, f"{pdf_name}_docling_indexed_text.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.document._export_to_indented_text())
        else:
            raise ValueError(f"Unsupported output format '{output_format}'.")

    def export_tables_from_pdf(self, pdf_path, output_folder, export_format="csv", mode=None, verbose=False):
        """
        Extracts and exports tables from a PDF to the specified format.

        Args:
            pdf_path (str): The path to the input PDF file.
            output_folder (str): The directory where the exported tables will be saved.
            export_format (str, optional): The format for exported tables. Options include:
                - "csv" (default)
                - "html"
                - "json"
                - "markdown"
            mode (str, optional): The table extraction mode. Options are "accurate" or None (default).
            verbose (bool, optional): If True, prints timing and progress information.
        
        Raises:
            ValueError: If the specified export format is unsupported.
        """
        supported_formats = {"csv", "html", "json", "markdown"}
        if export_format not in supported_formats:
            raise ValueError(f"Unsupported export format '{export_format}'. Supported formats: {supported_formats}")

        input_doc_path = Path(pdf_path)
        output_dir = Path(output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        if mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        start_time = time.time()
        conv_res = converter.convert(input_doc_path)
        doc_filename = conv_res.input.file.stem

        # Process and export tables
        for table_ix, table in enumerate(conv_res.document.tables):
            table_df: pd.DataFrame = table.export_to_dataframe()
            caption = table.caption_text(conv_res.document)
            element_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.{export_format}"

            if export_format == "csv":
                with element_filename.open("w", encoding="utf-8") as fp:
                    fp.write(f"# Caption: {caption}\n")
                    table_df.to_csv(fp, index=False)
            elif export_format == "html":
                raw_html = table.export_to_html()
                soup = BeautifulSoup(raw_html, "html.parser")
                caption_html = soup.new_tag("p")
                caption_html.string = f"Caption: {caption}"
                soup.insert(0, caption_html)
                with element_filename.open("w", encoding="utf-8") as fp:
                    fp.write(soup.prettify())
            elif export_format == "json":
                with element_filename.open("w", encoding="utf-8") as fp:
                    fp.write(f"// Caption: {caption}\n")
                    table_df.to_json(fp, orient="records", lines=True)
            elif export_format == "markdown":
                with element_filename.open("w", encoding="utf-8") as fp:
                    fp.write(f"**Caption:** {caption}\n\n")
                    fp.write(table_df.to_markdown(index=False))

        if verbose:
            print(f"Document converted and tables exported in {time.time() - start_time:.2f} seconds.")
            
    def extract_images(self, pdf_path, output_folder, verbose=False,export_pages = True, export_figures = True, export_tables = True):
        """
        Extracts images (pages, figures, tables) from a PDF and saves them to the specified folder.

        Args:
            pdf_path (str): The path to the input PDF file.
            output_folder (str): The directory where the extracted images will be saved.
            verbose (bool, optional): If True, prints timing and progress information.
            export_pages (bool, optional): If True, exports images of pages.
            export_figures (bool, optional): If True, exports images of figures.
            export_tables (bool, optional): If True, exports images of tables.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Configure pipeline options for image extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0  # Adjust image resolution scale
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        #pipeline_options.generate_table_images = True

        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        start_time = time.time()
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = doc_converter.convert(pdf_path)
        if verbose:
            print(f"Image extraction started for {pdf_path}...")

        pdf_name = Path(pdf_path).stem

        # Save page images
        if export_pages:
            for page_no, page in result.document.pages.items():
                page_image_filename = os.path.join(output_folder, f"{pdf_name}-page-{page_no}.png")
                with open(page_image_filename, "wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")

        # Save images of figures and tables
        table_counter = 0
        picture_counter = 0
        for element, _level in result.document.iterate_items():
            if isinstance(element, TableItem):
                if export_tables:
                    table_counter += 1
                    element_image_filename = os.path.join(output_folder, f"{pdf_name}-table-{table_counter}.png")
                    with open(element_image_filename, "wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")

            if isinstance(element, PictureItem):
                if export_figures:
                    picture_counter += 1
                    element_image_filename = os.path.join(output_folder, f"{pdf_name}-picture-{picture_counter}.png")
                    with open(element_image_filename, "wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")

        if verbose:
            print(f"Images extracted and saved to {output_folder} in {time.time() - start_time:.2f} seconds.")
            
if __name__ == "__main__":
    # use only the markdown format
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output/hard"
    doc_handler = DocumentHandler()
    """doc_handler.docling_serialize(pdf_path, output_dir, mode=None, output_format="markdown",verbose=True)
    #also export the tables
    doc_handler.export_tables_from_pdf(pdf_path, output_dir, export_format="markdown", mode=None, verbose=True)
    
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard_multicolumns.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output/hard_multicolumns"
    doc_handler.docling_serialize(pdf_path, output_dir, mode=None, output_format="markdown",verbose=True)
    #also export the tables
    doc_handler.export_tables_from_pdf(pdf_path, output_dir, export_format="markdown", mode=None, verbose=True)

    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard_image_and_many.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output/hard_image_and_many"
    doc_handler.docling_serialize(pdf_path, output_dir, mode=None, output_format="markdown",verbose=True)
    #also export the tables
    doc_handler.export_tables_from_pdf(pdf_path, output_dir, export_format="markdown", mode=None, verbose=True)
    
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/cs_ai_2023_pdfs/hard_confusing.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/utils/output/hard_confusing"
    doc_handler.docling_serialize(pdf_path, output_dir, mode=None, output_format="markdown",verbose=True)
    #also export the tables
    doc_handler.export_tables_from_pdf(pdf_path, output_dir, export_format="markdown", mode=None, verbose=True)
    """
    #doc_handler.extract_images(pdf_path, output_dir, verbose=True,export_pages=False, export_figures=True, export_tables=False)
    doc_handler.export_tables_from_pdf(pdf_path, output_dir, export_format="markdown", mode=None, verbose=True)
