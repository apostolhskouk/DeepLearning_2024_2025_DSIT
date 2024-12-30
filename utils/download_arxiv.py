import json
import re
import os
import requests

class ArxivPaperFilterDownloader:
    def __init__(self, metadata_file, output_dir):
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_comments(self, comments):
        """
        Extracts the number of pages and figures from the comments field.
        """
        pages = None
        figures = None
        if comments:
            # Extract number of pages (e.g., "37 pages", "7 Pages")
            pages_match = re.search(r"(\d+)\s*[Pp]ages?", comments)
            if pages_match:
                pages = int(pages_match.group(1))

            # Extract number of figures (e.g., "14 figures", "19 Figures")
            figures_match = re.search(r"(\d+)\s*[Ff]igures?", comments)
            if figures_match:
                figures = int(figures_match.group(1))
        return pages, figures

    def filter_papers(self, output_file, category, year, max_pages=None, min_figures=None, limit=-1):
        """
        Filters papers from the metadata file based on the specified criteria.

        Parameters:
            output_file (str): Path to save the filtered paper IDs.
            category (str): The category to filter papers by (e.g., 'cs.AI').
            year (str): The year to filter papers by (e.g., '2023').
            max_pages (int): Maximum number of pages allowed. If None, no restriction.
            min_figures (int): Minimum number of figures required. If None, no restriction.
            limit (int): Maximum number of results to include in the output. If -1, no limit.

        Returns:
            int: The number of papers that matched the criteria.
        """
        count = 0
        with open(self.metadata_file, "r") as f, open(output_file, "w") as out_f:
            for line in f:
                try:
                    paper = json.loads(line)
                    # Check if the paper matches the specified category
                    if category in paper.get("categories", ""):
                        # Check if the year exists anywhere in the versions list
                        versions_str = str(paper.get("versions", ""))
                        if year in versions_str:
                            # Parse the comments field for pages and figures
                            comments = paper.get("comments", "")
                            pages, figures = self.parse_comments(comments)

                            # Apply additional filters for pages and figures
                            if (max_pages is None or (pages is not None and pages <= max_pages)) and \
                               (min_figures is None or (figures is not None and figures >= min_figures)):
                                out_f.write(f"{paper['id']}\n")
                                count += 1

                                # Stop if the limit is reached
                                if limit != -1 and count >= limit:
                                    break
                except (KeyError, json.JSONDecodeError):
                    continue  # Skip any malformed entries

        print(f"Found {count} papers in category '{category}' from {year} matching the criteria.")
        return count

    def download_pdfs(self, id_file):
        """
        Downloads PDFs for the IDs listed in the specified file.

        Parameters:
            id_file (str): Path to the file containing paper IDs.
        """
        with open(id_file, "r") as f:
            for paper_id in f:
                paper_id = paper_id.strip()
                if paper_id:
                    print(f"Downloading PDF for ID: {paper_id}")
                    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                    output_path = os.path.join(self.output_dir, f"{paper_id}.pdf")

                    try:
                        response = requests.get(pdf_url)
                        if response.status_code == 200:
                            with open(output_path, "wb") as pdf_file:
                                pdf_file.write(response.content)
                            print(f"Downloaded: {output_path}")
                        else:
                            print(f"Failed to download PDF for ID: {paper_id}")
                    except requests.RequestException as e:
                        print(f"Error downloading PDF for ID: {paper_id} - {e}")

# Example usage
if __name__ == "__main__":
    
    ######################################################
    
    # The arxiv-metadata.json file can be downloaded from:
    #################### https://www.kaggle.com/datasets/Cornell-University/arxiv/data ####################
    
    ######################################################
    metadata_file_path = "./arxiv-metadata.json"
    output_dir_path = "./cs_ai_2023_pdfs"
    output_file_path = "./cs_ai_2023_ids.txt"
    
    category_filter = "cs.AI"
    year_filter = "2023"
    max_pages_filter = 15  # Maximum 15 pages
    min_figures_filter = 1  # Minimum 1 figure
    result_limit = 100  # Limit to 100 results

    downloader = ArxivPaperFilterDownloader(metadata_file_path, output_dir_path)
    downloader.filter_papers(output_file_path, category_filter, year_filter, max_pages_filter, min_figures_filter, result_limit)
    downloader.download_pdfs(output_file_path)