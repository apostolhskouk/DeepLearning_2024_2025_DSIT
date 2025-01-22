import os
import time
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import PyPDF2
import re
import easyocr
from bs4 import BeautifulSoup
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import PictureItem, TableItem

class DocumentHandler:
    """
    A class for handling document processing tasks, including converting PDFs to various formats
    and exporting tables from PDFs to structured formats.
    """

    def __init__(self):
        """
        Initializes the DocumentHandler instance with a DocumentConverter and EasyOCR reader.
        """
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)

    # --- Μέθοδοι του Πρώτου Κώδικα ---
    
    def docling_serialize(self, pdf_path, output_folder, mode=None, output_format="markdown", verbose=False, do_ocr=True, do_table_structure=True):
        """
        Converts a PDF to various output formats using Docling.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        if mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
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
        return output_file

    def export_tables_from_pdf(self, pdf_path, output_folder, export_format="csv", mode=None, verbose=False, do_ocr=True, do_table_structure=True):
        """
        Extracts and exports tables from a PDF to the specified format.
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
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
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

    # --- Μέθοδοι του Δεύτερου Κώδικα Ενσωματωμένοι ως Μέθοδοι της Κλάσης ---

    def extract_text_from_pdf_method(self, pdf_path):
        """
        Extracts all text from a PDF file.
        """
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text

    def extract_ids_from_text_method(self, pdf_text):
        """
        Extracts figure and table IDs from the extracted PDF text.
        """
        figure_pattern = re.compile(r'(Figure|Fig\.?)\s?(\d+(\.\d+)?)', re.IGNORECASE)
        table_pattern = re.compile(r'(Table)\s?(\d+(\.\d+)?)', re.IGNORECASE)

        figure_ids = [f"Figure {match[1]}" for match in figure_pattern.findall(pdf_text)]
        table_ids = [f"Table {match[1]}" for match in table_pattern.findall(pdf_text)]

        return figure_ids, table_ids

    def get_items_in_page_method(self, doc, page_no):
        """
        Retrieves all items (tables and figures) present in a specific page of the document.
        """
        items = []
        for element, _level in doc.iterate_items():
            if element.prov and len(element.prov) > 0:
                if element.prov[0].page_no == page_no:
                    items.append(element)
        return items

    def extract_bounding_boxes(self, pdf_path, output_folder="output_images_bounding_boxes", do_ocr=True, do_table_structure=True, 
                               export_pages=True, export_figures=True, export_tables=True, verbose=False):
        """
        Extracts images with bounding boxes for tables and figures and saves the data as JSON.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Δεν βρέθηκε το PDF: {pdf_path}")

        os.makedirs(output_folder, exist_ok=True)

        # Extract text and retrieve IDs
        pdf_text = self.extract_text_from_pdf_method(pdf_path)
        figure_ids, table_ids = self.extract_ids_from_text_method(pdf_text)

        if verbose:
            print(f"Extracted {len(figure_ids)} figure IDs and {len(table_ids)} table IDs.")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.generate_page_images = export_pages
        pipeline_options.generate_picture_images = export_figures
        pipeline_options.generate_table_images = export_tables

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        result = converter.convert(pdf_path)
        doc = result.document
        pdf_name = pdf_path.stem

        extracted_data = {}
        figure_counter = 0
        table_counter = 0

        for page_no, page_obj in doc.pages.items():
            if not page_obj.image or not page_obj.image.pil_image:
                if verbose:
                    print(f"No page image found for page {page_no}.")
                continue

            pil_im = page_obj.image.pil_image.copy()
            draw = ImageDraw.Draw(pil_im)

            img_width, img_height = pil_im.size
            pdf_width, pdf_height = page_obj.size.width, page_obj.size.height

            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            scale_factor = min(scale_x, scale_y)
            margin_x = (img_width - (pdf_width * scale_factor)) / 2
            margin_y = (img_height - (pdf_height * scale_factor)) / 2
            extracted_data[page_no] = []

            items_in_page = self.get_items_in_page_method(doc, page_no)
            if verbose:
                print(f"Found {len(items_in_page)} items in page {page_no}.")

            for item in items_in_page:
                if isinstance(item, (TableItem, PictureItem)):
                    if item.prov and item.prov[0].bbox:
                        l, t, r, b = item.prov[0].bbox.as_tuple()

                        l *= scale_x
                        r *= scale_x
                        t = pdf_height - t
                        b = pdf_height - b
                        t *= scale_y
                        b *= scale_y
                        t, b = min(t, b), max(t, b)

                        l = max(0, min(l, img_width))
                        r = max(0, min(r, img_width))
                        t = max(0, min(t, img_height))
                        b = max(0, min(b, img_height))

                        item_type = "table" if isinstance(item, TableItem) else "figure"

                        # Assign ID based on detected order
                        if item_type == "figure":
                            figure_counter += 1
                            if figure_counter <= len(figure_ids):
                                item_id = figure_ids[figure_counter - 1]
                            else:
                                item_id = f"Figure {figure_counter}"
                        else:
                            table_counter += 1
                            if table_counter <= len(table_ids):
                                item_id = table_ids[table_counter - 1]
                            else:
                                item_id = f"Table {table_counter}"

                        extracted_data[page_no].append({
                            "id": item_id,
                            "type": item_type,
                            "bbox": [l, t, r, b]
                        })

                        color = "red" if item_type == "table" else "blue"
                        draw.rectangle([(l, t), (r, b)], outline=color, width=3)

                        if verbose:
                            print(f"Processed {item_type} '{item_id}' on page {page_no} with bbox {l, t, r, b}.")

            out_image_path = Path(output_folder) / f"{pdf_name}_page_{page_no}.png"
            pil_im.save(out_image_path, "PNG")
            if verbose:
                print(f"Saved image with bounding boxes: {out_image_path}")

        json_output_path = Path(output_folder) / f"{pdf_name}_bbox_data.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4)

        if verbose:
            print(f"Bounding box data with IDs saved to {json_output_path}")

    # --- Υπόλοιπες Μέθοδοι Παραμένουν Όπως Είναι ---

    def _wrap_text_by_width(self, text: str, max_width: int, font: ImageFont.ImageFont) -> list[str]:
        """
        Splits `text` into multiple lines, ensuring each line's width 
        does not exceed `max_width`.
        """
        lines = []
        words = text.split()
        if not words:
            return [text]

        current_line = words[0]
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            w, _ = font.getbbox(test_line)[2:]  
            if w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return lines

    def _embed_caption_in_image(self, image: Image.Image, caption: str) -> Image.Image:
        """
        Create a new image with multi-line caption text drawn at the bottom, 
        wrapping text so it doesn't exceed the original image's width.
        """
        caption = caption.strip()
        if not caption:
            return image

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except OSError:
            font = ImageFont.load_default()

        # We'll wrap lines so that none exceed the original image's width.
        max_line_width = image.width
        wrapped_lines = self._wrap_text_by_width(caption, max_line_width, font)

        # We'll need to measure each line's height:
        line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in wrapped_lines]
        total_text_height = sum(line_heights)

        # Some padding around:
        padding = 10
        new_width = image.width + (padding * 2)
        new_height = image.height + total_text_height + (padding * 2)

        # Create the new extended image (white background)
        new_img = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))
        # Paste the original image in the top-center
        new_img.paste(image, (padding, 0))

        # Draw each wrapped line near the bottom
        draw_new = ImageDraw.Draw(new_img)
        current_y = image.height + padding

        for line in wrapped_lines:
            line_width = font.getbbox(line)[2]  # Measure line width
            line_x = padding + (image.width - line_width) // 2  # center the text
            draw_new.text((line_x, current_y), line, fill=(0, 0, 0), font=font)
            current_y += font.getbbox(line)[3] - font.getbbox(line)[1]  # Move down by line height

        return new_img

    def _is_relevant_image(self, image: Image.Image, caption: str) -> bool:
        """
        Uses EasyOCR to determine if an image is relevant based on the presence of meaningful text.
        
        Args:
            image (PIL.Image.Image): The image to analyze.
            caption (str): The caption associated with the image.
        
        Returns:
            bool: True if the image contains relevant text, False otherwise.
        """
        # Convert PIL image to OpenCV-compatible format (np.array)
        image_np = np.array(image)

        # Run EasyOCR on the image
        ocr_results = self.ocr_reader.readtext(image_np)
        extracted_text = " ".join([text for (_, text, _) in ocr_results])

        # Basic filtering: Check if meaningful text is present
        relevant_keywords = ["figure", "fig", "table", "graph", "data"]
        if any(keyword.lower() in caption.lower() for keyword in relevant_keywords):
            return True
        elif any(keyword.lower() in extracted_text.lower() for keyword in relevant_keywords):
            return True

        # Alternatively, check the text length
        if len(extracted_text.strip()) > 10:  # Arbitrary threshold
            return True

        return False

    def _is_reference_table(self, ocr_result):
        citation_pattern = re.compile(r'\[\d+\]')
        citation_count = sum(1 for _, text, _ in ocr_result if citation_pattern.search(text))

        # Heuristic: If there are multiple citations, it's likely a reference table
        return citation_count > 5

    # --- Μέθοδοι του Δεύτερου Κώδικα ---

    def extract_text_from_pdf_second_code(self, pdf_path):
        """
        Extracts all text from a PDF file.
        """
        return self.extract_text_from_pdf_method(pdf_path)

    def extract_ids_from_text_second_code(self, pdf_text):
        """
        Extracts figure and table IDs from the extracted PDF text.
        """
        return self.extract_ids_from_text_method(pdf_text)

    def get_items_in_page_second_code(self, doc, page_no):
        """
        Retrieves all items (tables and figures) present in a specific page of the document.
        """
        return self.get_items_in_page_method(doc, page_no)

    def extract_images_second_code(self, pdf_path, output_folder, verbose=False, export_pages=True, export_figures=True, export_tables=True, do_ocr=True, do_table_structure=True):
        """
        Extracts images using the second code's method.
        """
        self.extract_bounding_boxes(
            pdf_path=pdf_path,
            output_folder=output_folder,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            export_pages=export_pages,
            export_figures=export_figures,
            export_tables=export_tables,
            verbose=verbose
        )

if __name__ == "__main__":
    # Παράδειγμα χρήσης του συνδυασμένου κώδικα
    pdf_path = r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\2208.08160.pdf"
    output_dir_images = r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\output_images"
    output_dir_json = r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\output_json"

    doc_handler = DocumentHandler()

    # Δημιουργία ξεχωριστών φακέλων για εικόνες και JSON
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_json, exist_ok=True)

    # Εκτέλεση της πρώτης λειτουργίας (εξαγωγή εικόνων)
    print("Ξεκινώντας εξαγωγή εικόνων από τον πρώτο κώδικα...")
    doc_handler.export_tables_from_pdf(
        pdf_path=pdf_path,
        output_folder=output_dir_images,
        export_format="markdown",
        mode=None,
        verbose=True,
        do_ocr=False,  # Προσαρμόστε ανάλογα με τις ανάγκες σας
        do_table_structure=False  # Προσαρμόστε ανάλογα με τις ανάγκες σας
    )

    # Εκτέλεση της δεύτερης λειτουργίας (εξαγωγή bounding boxes και JSON)
    print("\nΞεκινώντας εξαγωγή bounding boxes και JSON από τον δεύτερο κώδικα...")
    doc_handler.extract_bounding_boxes(
        pdf_path=pdf_path,
        output_folder=output_dir_json,
        do_ocr=True,
        do_table_structure=True,
        export_pages=True,
        export_figures=True,
        export_tables=True,
        verbose=True
    )

    print("\nΟλοκλήρωση διαδικασίας.")
