import os
import time
import json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
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
    and exporting tables and figures from PDFs to structured formats.
    """

    def __init__(self):
        """
        Initializes the DocumentHandler instance with a DocumentConverter and EasyOCR reader.
        """
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)

    
    def extract_bounding_boxes(self, pdf_path, output_folder="output_images_bounding_boxes", do_ocr=True, do_table_structure=True, 
                           export_pages=True, export_figures=True, export_tables=True, verbose=False):
        """
        Extracts images with bounding boxes for tables and figures and saves the data as JSON.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        os.makedirs(output_folder, exist_ok=True)

        # Extract text using Docling
        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            generate_page_images=True  
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        conv_res = converter.convert(pdf_path)
        doc = conv_res.document

        if verbose:
            print(f"Extracted text from PDF.")

        pdf_name = pdf_path.stem

        # Extract image files and sort them by numerical order
        def extract_number(filename):
            match = re.search(r'-(\d+)', filename.stem)
            return int(match.group(1)) if match else float('inf')

        image_folder = Path(r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\output_images")
        image_files = sorted(image_folder.glob(f"{pdf_name}-*.png"), key=extract_number)

        if not image_files:
            print("Error: No image files found in the specified folder.")
            return

        # Mapping of image filenames to their types
        image_mapping = {img.stem: "table" if "table" in img.stem.lower() else "figure" for img in image_files}
        
        extracted_data = {}
        used_images = set()  # To avoid repeated usage of images

        for page_no, page_obj in doc.pages.items():
            if not page_obj.image or not page_obj.image.pil_image:
                if verbose:
                    print(f"Warning: No page image found for page {page_no}. Skipping...")
                continue

            pil_im = page_obj.image.pil_image.copy()
            draw = ImageDraw.Draw(pil_im)

            img_width, img_height = pil_im.size
            pdf_width, pdf_height = page_obj.size.width, page_obj.size.height

            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            extracted_data[page_no] = []

            items_in_page = self.get_items_in_page_method(doc, page_no)
            if verbose:
                print(f"Page {page_no}: Found {len(items_in_page)} items.")

            for item in items_in_page:
                if isinstance(item, (TableItem, PictureItem)):
                    if item.prov and item.prov[0].bbox:
                        l, t, r, b = item.prov[0].bbox.as_tuple()

                        # Convert PDF coordinates to image coordinates
                        l *= scale_x
                        r *= scale_x
                        t = pdf_height - t
                        b = pdf_height - b
                        t *= scale_y
                        b *= scale_y
                        t, b = min(t, b), max(t, b)

                        # Ensure bounding boxes are within image dimensions
                        l = max(0, min(l, img_width - 1))
                        r = max(0, min(r, img_width - 1))
                        t = max(0, min(t, img_height - 1))
                        b = max(0, min(b, img_height - 1))

                        # Assign ID from available images, avoiding duplication
                        for img_name in image_mapping:
                            if img_name not in used_images:
                                item_id = img_name
                                item_type = image_mapping[img_name]
                                used_images.add(img_name)
                                break
                        else:
                            continue  # Skip if all images used

                        extracted_data[page_no].append({
                            "id": item_id,
                            "type": item_type,
                            "bbox": [l, t, r, b]
                        })

                        color = "red" if item_type == "table" else "blue"
                        draw.rectangle([(l, t), (r, b)], outline=color, width=3)

                        if verbose:
                            print(f"Processed '{item_id}' ({item_type}) on page {page_no} with bbox {l, t, r, b}.")

            # Save the processed image with bounding boxes
            out_image_path = Path(output_folder) / f"{pdf_name}_page_{page_no}.png"
            pil_im.save(out_image_path, "PNG")
            if verbose:
                print(f"Saved image with bounding boxes: {out_image_path}")

        # Save corrected JSON
        json_output_path = Path(output_folder) / f"{pdf_name}_bbox_data.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4)

        if verbose:
            print(f"Bounding box data saved to {json_output_path}")






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

        # Wrap lines so that none exceed the original image's width.
        max_line_width = image.width
        wrapped_lines = self._wrap_text_by_width(caption, max_line_width, font)

        # Measure each line's height
        line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in wrapped_lines]
        total_text_height = sum(line_heights)

        # Padding around text
        padding = 10
        new_width = image.width + (padding * 2)
        new_height = image.height + total_text_height + (padding * 2)

        # Create the new extended image (white background)
        new_img = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))
        # Paste the original image at the top-center
        new_img.paste(image, (padding, 0))

        # Draw each wrapped line near the bottom
        draw_new = ImageDraw.Draw(new_img)
        current_y = image.height + padding

        for line in wrapped_lines:
            line_width = font.getbbox(line)[2]  # Measure line width
            line_x = padding + (image.width - line_width) // 2  # Center the text
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

    

    def extract_images_first_code(self, pdf_path, output_folder, verbose=False, export_pages=True, export_figures=True, export_tables=True, add_caption=True, filter_irrelevant=True, do_ocr=True, do_table_structure=True):
        """
        Wrapper method to call the first code's extract_images method.
        """
        self.extract_images(
            pdf_path=pdf_path,
            output_folder=output_folder,
            verbose=verbose,
            export_pages=export_pages,
            export_figures=export_figures,
            export_tables=export_tables,
            add_caption=add_caption,
            filter_irrelevant=filter_irrelevant,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure
        )

    def extract_bounding_boxes_second_code(self, pdf_path, output_folder, verbose=False, export_pages=True, export_figures=True, export_tables=True, do_ocr=True, do_table_structure=True):
        """
        Wrapper method to call the second code's extract_bounding_boxes method.
        """
        self.extract_bounding_boxes(
            pdf_path=pdf_path,
            output_folder=output_folder,
            verbose=verbose,
            export_pages=export_pages,
            export_figures=export_figures,
            export_tables=export_tables,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure
        )

    
    def extract_images(self, pdf_path, output_folder="output_images", verbose=False, export_pages=True, export_figures=True, export_tables=True, add_caption=True, filter_irrelevant=True, do_ocr=True, do_table_structure=True):
        """
        Extracts images (pages, figures, tables) from a PDF and saves them to 
        the specified folder. Also embeds each image’s caption as multiline 
        text if needed.

        Args:
            pdf_path (str): The path to the input PDF file.
            output_folder (str): The directory where the extracted images will be saved.
            verbose (bool, optional): If True, prints timing and progress information.
            export_pages (bool, optional): If True, exports images of pages.
            export_figures (bool, optional): If True, exports images of figures.
            export_tables (bool, optional): If True, exports images of tables.
            do_ocr (bool, optional): If True, triggers OCR on the PDF if needed.
            do_table_structure (bool, optional): If True, tries to extract structural table info.
            add_caption (bool, optional): If True, adds captions to images.
            filter_irrelevant (bool, optional): If True, filters out irrelevant images based on OCR.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Configure pipeline options for image extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = export_pages
        pipeline_options.generate_picture_images = export_figures
        pipeline_options.do_ocr = do_ocr
        pipeline_options.do_table_structure = do_table_structure
        pipeline_options.generate_table_images = export_tables

        start_time = time.time()
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(pdf_path)
        if verbose:
            print(f"Image extraction started for {pdf_path}...")

        pdf_name = Path(pdf_path).stem

        # Save page images
        if export_pages:
            for page_no, page in result.document.pages.items():
                if page.image and page.image.pil_image:
                    page_image_filename = os.path.join(
                        output_folder, f"{pdf_name}-page-{page_no}.png"
                    )
                    page.image.pil_image.save(page_image_filename, "PNG")
                    if verbose:
                        print(f"Saved page image: {page_image_filename}")

        # Save images of figures and tables
        table_counter = 0
        picture_counter = 0
        for element, _level in result.document.iterate_items():
            # For TableItem
            if isinstance(element, TableItem) and export_tables:
                table_counter += 1
                if verbose:
                    print(f"Processing Table {table_counter}")
                if not add_caption:
                    element_image_filename = os.path.join(output_folder, f"{pdf_name}-table-{table_counter}.png")
                    with open(element_image_filename, "wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")
                    if verbose:
                        print(f"Saved table image without caption: {element_image_filename}")
                    continue
                table_caption = element.caption_text(result.document).strip()
                table_img = element.get_image(result.document)
                if table_img:
                    # Docling for extracting text
                    ocr_result = self.ocr_reader.readtext(np.array(table_img))

                    if (table_caption == "" or table_caption is None) and self._is_reference_table(ocr_result):
                        if verbose:
                            print("Skipping reference table.")
                        continue
                    table_img_with_cap = self._embed_caption_in_image(table_img, table_caption)
                    element_image_filename = os.path.join(
                        output_folder, f"{pdf_name}-table-{table_counter}.png"
                    )
                    with open(element_image_filename, "wb") as fp:
                        table_img_with_cap.save(fp, "PNG")
                    if verbose:
                        print(f"Saved table image with caption: {element_image_filename}")

            # For PictureItem
            elif isinstance(element, PictureItem) and export_figures:
                picture_counter += 1
                if verbose:
                    print(f"Processing Picture {picture_counter}")
                if not add_caption:
                    element_image_filename = os.path.join(output_folder, f"{pdf_name}-picture-{picture_counter}.png")
                    with open(element_image_filename, "wb") as fp:
                        element.get_image(result.document).save(fp, "PNG")
                    if verbose:
                        print(f"Saved picture image without caption: {element_image_filename}")
                    continue
                figure_caption = element.caption_text(result.document).strip()
                figure_img = element.get_image(result.document)
                if figure_img:
                    if filter_irrelevant and not self._is_relevant_image(figure_img, figure_caption):
                        if verbose:
                            print("Skipping irrelevant image.")
                        continue                    
                    fig_img_with_cap = self._embed_caption_in_image(figure_img, figure_caption)
                    element_image_filename = os.path.join(
                        output_folder, f"{pdf_name}-picture-{picture_counter}.png"
                    )
                    with open(element_image_filename, "wb") as fp:
                        fig_img_with_cap.save(fp, "PNG")
                    if verbose:
                        print(f"Saved picture image with caption: {element_image_filename}")

        if verbose:
            print(
                f"Images extracted and saved to {output_folder} "
                f"in {time.time() - start_time:.2f} seconds."
            )

# main

if __name__ == "__main__":
    
    pdf_path = r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\2208.00808.pdf"
    
    
    output_dir_images = r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\output_images"
    output_dir_json = r"C:\Users\petro\OneDrive\Desktop\DeepLearning_2024_2025_DSIT-main\cs_ai_2023_pdfs\output_json"

    doc_handler = DocumentHandler()

    # Seperate files for images and JSON
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_json, exist_ok=True)

    # Extract output images
    print("Extracting images")
    doc_handler.extract_images(
        pdf_path=pdf_path,
        output_folder=output_dir_images,
        verbose=True,
        export_pages=False,      
        export_figures=True,
        export_tables=True,
        add_caption=True,
        filter_irrelevant=True,
        do_ocr=True,
        do_table_structure=True
)


    # Extract bounding boxes and JSON
    print("\nStarting the extraction for bounding boxes and JSON ")
    doc_handler.extract_bounding_boxes(
    pdf_path=pdf_path,
    output_folder=output_dir_json,
    verbose=True,
    export_pages=True,
    export_figures=True,
    export_tables=True,
    do_ocr=True,
    do_table_structure=True
)

