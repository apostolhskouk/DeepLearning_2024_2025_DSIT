import os
import time
from pathlib import Path
from PIL import Image

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TableItem, PictureItem

# Hugging Face/Donut imports
import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor, pipeline

class DocumentHandler:
    """
    Κλάση για εξαγωγή εικόνων ΜΟΝΟ από tables και figures ενός PDF μέσω Docling.
    """

    def extract_table_figure_images(self, pdf_path, output_folder):
        """
        Εξάγει ΕΙΚΟΝΕΣ μόνο από tables και figures και τις αποθηκεύει ως PNG.
        Επιστρέφει λίστα με τα paths των δημιουργημένων εικόνων.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Ρυθμίσεις Docling: ΜΟΝΟ tables και figures, άρα δεν παράγουμε σελίδες
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = False       # Δεν αποθηκεύουμε σελίδες
        pipeline_options.generate_picture_images = True     # Figures
        pipeline_options.generate_table_images = True       # Tables
        pipeline_options.do_ocr = False                     # Δεν κάνουμε OCR
        pipeline_options.do_table_structure = False         # Δεν κάνουμε ανάλυση πίνακα, μόνο εικόνα

        start_time = time.time()
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(pdf_path)
        pdf_name = Path(pdf_path).stem
        
        if not result or not result.document:
            print("Η μετατροπή απέτυχε ή δεν βρέθηκε έγκυρο PDF.")
            return []

        extracted_paths = []
        table_counter = 0
        figure_counter = 0

        # Περιηγούμαστε σε όλα τα στοιχεία (items) του εγγράφου
        for element, _ in result.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                table_img_filename = os.path.join(output_folder, f"{pdf_name}-table-{table_counter}.png")
                table_img = element.get_image(result.document)  # PIL Image
                if table_img:
                    table_img.save(table_img_filename, "PNG")
                    extracted_paths.append(table_img_filename)
                    
            elif isinstance(element, PictureItem):
                figure_counter += 1
                figure_img_filename = os.path.join(output_folder, f"{pdf_name}-figure-{figure_counter}.png")
                figure_img = element.get_image(result.document)  # PIL Image
                if figure_img:
                    figure_img.save(figure_img_filename, "PNG")
                    extracted_paths.append(figure_img_filename)

        print(f"Εξαγωγή μόνο tables/figures ολοκληρώθηκε σε {time.time() - start_time:.2f} δευτερόλεπτα.")
        return extracted_paths


class DonutSummarizer:
    """
    Χρησιμοποιεί το μοντέλο Donut (OCR-free) για εξαγωγή κειμένου από εικόνες
    και Summarization του κειμένου που εξάγεται.
    """

    def __init__(self,
                 donut_model_name="naver-clova-ix/donut-base",
                 summarization_model_name="facebook/bart-large-cnn"):
        """
        Φορτώνει τα δύο μοντέλα:
        - Donut (VisionEncoderDecoder) για OCR-free text extraction
        - Summarization model (π.χ. BART-large-cnn)
        """
        # Donut
        self.processor = AutoProcessor.from_pretrained(donut_model_name)
        self.donut_model = VisionEncoderDecoderModel.from_pretrained(donut_model_name)
        self.donut_model.eval()

        # Summarizer
        self.summarizer = pipeline("summarization", model=summarization_model_name)

    def extract_text_from_image(self, image: Image.Image, max_length=512) -> str:
        """
        Εξάγει κείμενο από μία εικόνα χρησιμοποιώντας το Donut μοντέλο (χωρίς OCR).
        """
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            output_ids = self.donut_model.generate(pixel_values, max_length=max_length)
        raw_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return raw_text

    def summarize_text(self, text: str, max_len: int = 15, min_len: int = 5) -> str:
        """
        Εφαρμόζει summarization στο κείμενο που επέστρεψε το Donut.
        """
        summary_list = self.summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary_list[0]["summary_text"]

    def summarize_image(self, image: Image.Image) -> str:
        """
        Ενιαία ροή:
         1) Εξάγει κείμενο με το Donut.
         2) Κάνει summarization στο κείμενο.
        """
        extracted_text = self.extract_text_from_image(image)
        return self.summarize_text(extracted_text)


def main():
    # Παράδειγμα μονοπατιών
    pdf_path = "C:/Users/petro/cs_ai_2023_pdfs/2008.01302.pdf"
    output_folder = "C:/Users/petro/cs_ai_2023_pdfs/output/2008.01302"

    # 1) Εξάγουμε μόνο tables/figures από το PDF
    doc_handler = DocumentHandler()
    extracted_image_paths = doc_handler.extract_table_figure_images(pdf_path, output_folder)

    # 2) Φορτώνουμε τον DonutSummarizer για OCR-free summarization
    donut_summarizer = DonutSummarizer(
        donut_model_name="naver-clova-ix/donut-base",
        summarization_model_name="facebook/bart-large-cnn"
    )

    # 3) Για κάθε εικόνα (πίνακας/figure), κάνουμε συνοπτική παρουσίαση
    for img_path in extracted_image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            summary = donut_summarizer.summarize_image(image)
            print(f"Summary for {os.path.basename(img_path)}:\n{summary}\n")
        except Exception as e:
            print(f"Σφάλμα με την εικόνα {img_path}: {e}")


if __name__ == "__main__":
    main()
