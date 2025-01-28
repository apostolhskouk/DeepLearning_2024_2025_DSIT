import os
import time
import json
from pathlib import Path
import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import fitz
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from bs4 import BeautifulSoup
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from textwrap import wrap
import easyocr
import re
import tempfile
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re
from datetime import datetime
from docling.chunking import HybridChunker

class DocumentHandler:
    """
    A class for handling document processing tasks, including converting PDFs to various formats
    and exporting tables from PDFs to structured formats.
    """

    def __init__(self,use_gpu = True):
        """
        Initializes the DocumentHandler instance with a DocumentConverter.
        """
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)

    def docling_serialize(self, pdf_path, output_folder, mode=None, output_format="markdown", verbose=False,do_ocr = True, do_table_structure = True,strict_text = False,image_placeholder: str = "<!-- image -->"):
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
                f.write(result.document.export_to_markdown(strict_text=strict_text,image_placeholder=image_placeholder))
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

    def export_tables_from_pdf(self, pdf_path, output_folder, export_format="csv", mode=None, verbose=False,do_ocr = True, do_table_structure = True):
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

    def _embed_caption_in_image(self, image: PIL.Image.Image, caption: str) -> PIL.Image.Image:
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

    def _is_relevant_image(self, image: PIL.Image.Image, caption: str) -> bool:
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
        # Patterns to detect citations
        citation_pattern = re.compile(r'\[\w+[\+\-\.]?\w*\]')  # Matches Alphanumeric Citations ([1], [A+15], etc.)
        
        # Pattern to detect reference headers
        reference_header_pattern = re.compile(r'(References|Bibliography|Citations)', re.IGNORECASE)
        
        # Pattern to detect common reference-related terms
        reference_context_pattern = re.compile(
            r'(et al\.|pp\.|vol\.|no\.|ed\.|journal|conference|proceedings|arxiv|preprint|doi|http|https|www\.)', 
            re.IGNORECASE
        )
        
        citation_count = 0
        has_reference_header = False
        has_reference_context = False

        for _, text, _ in ocr_result:
            # Count citations
            if citation_pattern.search(text):
                citation_count += 1
            
            # Check for reference headers
            if reference_header_pattern.search(text):
                has_reference_header = True
            
            # Check for reference context
            if reference_context_pattern.search(text):
                has_reference_context = True

        # Heuristic: If there are multiple citations, a reference header, or reference-related context, it's likely a reference table
        return (citation_count > 5) or has_reference_header or has_reference_context

    def extract_chunks(self, pdf_path, output_folder, verbose=False, mode="fast"):
        """
        Extracts text chunks from a PDF file and saves them as individual text files.
        """
        
        start_time = time.time()
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pipeline_options = PdfPipelineOptions(do_table_structure=True)
        if mode == "accurate":
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = doc_converter.convert(pdf_path)
        doc = result.document
        
        chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5")  # set tokenizer as needed
        chunk_iter = chunker.chunk(doc)
        chunk_list = list(chunk_iter)
        fileName = "docling-chunks.txt"
        chunk_filename = os.path.join(output_folder, fileName)
        for chunk_ix, chunk in enumerate(chunk_list):
            chunk_heading = chunk.meta.headings[0]
            chunk_text = chunk.text

            with open(chunk_filename, "a", encoding="utf-8") as f:
                f.write(f"Chunk {chunk_ix + 1}:\n")
                f.write(f"Heading: {chunk_heading}\n")
                f.write(chunk_text)
                f.write("\n\n")            

        if verbose:
            duration = time.time() - start_time
            print(f"Text chunks extracted in {duration:.2f} seconds")

    def extract_images(
        self,
        pdf_path,
        output_folder,
        verbose=False,
        export_pages=True,
        export_figures=True,
        export_tables=True,
        do_ocr=True,
        do_table_structure=True,
        add_caption=True,
        filter_irrelevant=True,
        generate_metadata=False,
        generate_annotated_pdf=False,
        generate_descriptions=False,
    ):
        """
        Extracts images from PDF with captions, metadata, and annotated PDF support.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if generate_descriptions:
        
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not hasattr(self, 'desc_model'):
                self.desc_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                ).to(device)
                self.desc_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
        metadata = []
        pdf_name = Path(pdf_path).stem
        annotated_pdf_path = os.path.join(output_folder, f"{pdf_name}-annotated.pdf") if generate_annotated_pdf else None

        # Configure processing pipeline
        pipeline_options = PdfPipelineOptions(
            images_scale=2.0,
            generate_page_images=export_pages,
            generate_picture_images=export_figures,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            generate_table_images=export_tables,
            do_picture_classification=True
        )

        start_time = time.time()
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = doc_converter.convert(pdf_path)
        if verbose:
            print(f"Processing PDF: {pdf_path}...")

        # Process pages
        if export_pages:
            for page_no, page in result.document.pages.items():
                if page.image and page.image.pil_image:
                    img_path = os.path.join(output_folder, f"{pdf_name}-page-{page_no}.png")
                    page.image.pil_image.save(img_path, "PNG")
                    
                    if generate_metadata:
                        scale = pipeline_options.images_scale
                        meta_entry = {
                            "filename": os.path.basename(img_path),
                            "type": "page",
                            "page_no": page_no,
                            "bbox": {
                                "l": 0.0,
                                "t": 0.0,
                                "r": page.image.pil_image.width / scale,
                                "b": page.image.pil_image.height / scale
                                #"coord_origin": "top-left"
                            }
                        }
                        metadata.append(meta_entry)

        # Process tables and figures
        table_counter, picture_counter = 0, 0
        for element, _ in result.document.iterate_items():
            # Table processing
            if isinstance(element, TableItem) and export_tables:
                table_counter += 1
                img_path = os.path.join(output_folder, f"{pdf_name}-table-{table_counter}.png")
                table_caption = element.caption_text(result.document).strip()
                
                # Process table image
                with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                    element.get_image(result.document).save(temp_file.name, "PNG")
                    if self._is_reference_table(self.ocr_reader.readtext(temp_file.name)):
                        if verbose: print("Skipping reference table")
                        continue
                    
                    final_img = self._embed_caption_in_image(Image.open(temp_file.name), table_caption) if add_caption else Image.open(temp_file.name)
                    final_img.save(img_path, "PNG")

                # Add metadata
                if generate_metadata:
                    prov = element.prov[0]
                    bbox = prov.bbox.to_top_left_origin(result.document.pages[prov.page_no].size.height)
                    if generate_descriptions:
                        try:
                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img_path},
                                    {"type": "text", "text": """
                                        Given this table from a scientific paper, provide a single technically precise sentence that:
                                        1. States the type of visualization
                                        2. Describes the main scientific concept or finding shown
                                        3. Mentions key variables or metrics involved
                                        Keep under 50 words with technical terms. Focus on core message.
                                    """}
                                ]
                            }]

                            text = self.desc_processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            image_inputs, video_inputs = process_vision_info(messages)

                            inputs = self.desc_processor(
                                text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                            ).to(device)

                            generated_ids = self.desc_model.generate(**inputs, max_new_tokens=128)
                            generated_ids_trimmed = [
                                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                            ]
                            desc = self.desc_processor.batch_decode(
                                generated_ids_trimmed, 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False
                            )[0]
                            
                        except Exception as e:
                            desc = "Description generation failed"
                    
                    metadata.append({
                        "filename": os.path.basename(img_path),
                        "type": "table",
                        "page_no": prov.page_no,
                        "bbox": bbox.model_dump(),
                        "caption": table_caption or None,
                        "description": desc.strip() if generate_descriptions else None
                    })

            # Figure processing
            elif isinstance(element, PictureItem) and export_figures:
                classification_data = element.annotations[0]
                picture_counter += 1
                img_path = os.path.join(output_folder, f"{pdf_name}-picture-{picture_counter}.png")
                figure_caption = element.caption_text(result.document).strip()
                
                if filter_irrelevant: 
                    if not self._is_relevant_image( element.get_image(result.document), figure_caption) or "natural_image" in classification_data.predicted_classes[0].class_name:
                        if verbose: 
                            print("Skipping irrelevant image")
                        continue
                
                # Save image with caption
                img = self._embed_caption_in_image(
                    element.get_image(result.document), 
                    figure_caption
                )
                img.save(img_path, "PNG")

                # Add metadata
                if generate_metadata:
                    prov = element.prov[0]
                    bbox = prov.bbox.to_top_left_origin(result.document.pages[prov.page_no].size.height)
                    if generate_descriptions:
                        try:
                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img_path},
                                    {"type": "text", "text": """
                                        Given this figure from a scientific paper, provide a single technically precise sentence that:
                                        1. States the type of visualization
                                        2. Describes the main scientific concept or finding shown
                                        3. Mentions key variables or metrics involved
                                        Keep under 50 words with technical terms. Focus on core message.
                                    """}
                                ]
                            }]

                            text = self.desc_processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            image_inputs, video_inputs = process_vision_info(messages)

                            inputs = self.desc_processor(
                                text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                            ).to(device)

                            generated_ids = self.desc_model.generate(**inputs, max_new_tokens=128)
                            generated_ids_trimmed = [
                                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                            ]
                            desc = self.desc_processor.batch_decode(
                                generated_ids_trimmed, 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False
                            )[0]
                            
                        except Exception as e:
                            desc = "Description generation failed"
                    metadata.append({
                        "filename": os.path.basename(img_path),
                        "type": "picture",
                        "page_no": prov.page_no,
                        "bbox": bbox.model_dump(),
                        "caption": figure_caption or None,
                        "description": desc.strip() if generate_descriptions else None
                    })

        # Generate metadata file
        if generate_metadata:
            metadata_path = os.path.join(output_folder, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=lambda x: x.model_dump() if hasattr(x, 'model_dump') else x)
            if verbose:
                print(f"Metadata saved to {metadata_path}")

        # Generate annotated PDF
        if generate_annotated_pdf:
            doc = fitz.open(pdf_path)
            annot_config = {
                "page": {"color": (1,0,0), "width": 1.5},
                "table": {"color": (0,0.5,0), "width": 1.2},
                "picture": {"color": (0,0,1), "width": 1.0}
            }
            
            for entry in metadata:
                page = doc[entry["page_no"] - 1]
                rect = fitz.Rect(entry["bbox"]["l"], entry["bbox"]["t"], entry["bbox"]["r"], entry["bbox"]["b"])
                annot = page.add_rect_annot(rect)
                style = annot_config[entry["type"]]
                annot.set_border(width=style["width"], dashes=[0])
                annot.set_colors(stroke=style["color"], fill=None)
                annot.set_opacity(0.7)
            
            doc.save(annotated_pdf_path)
            if verbose:
                print(f"Annotated PDF saved to {annotated_pdf_path}")

        if verbose:
            duration = time.time() - start_time
            print(f"Process completed in {duration:.2f} seconds")

        return {
            "image_count": table_counter + picture_counter,
            "metadata_path": os.path.join(output_folder, "metadata.json") if generate_metadata else None,
            "annotated_pdf_path": annotated_pdf_path,
            "descriptions_generated": generate_descriptions
        }

if __name__ == "__main__":
    pdf_path = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/demos/cs_ai_2024_pdfs/test2.pdf"
    output_dir = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/test"    
    
    doc_handler = DocumentHandler()
    doc_handler.extract_images(pdf_path,output_dir,verbose=True,export_pages=False,export_figures=True,export_tables=True,do_ocr=True,do_table_structure=True,add_caption=True,filter_irrelevant=True,generate_metadata=True,generate_annotated_pdf=True,generate_descriptions=True)
    # doc_handler.extract_chunks(pdf_path,output_dir,verbose=True)