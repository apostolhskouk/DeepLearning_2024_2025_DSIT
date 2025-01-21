from FlagEmbedding import BGEM3FlagModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from PDFSerlalizer import DocumentHandler
import re 
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class SemanticChunker:
    def __init__(self):
        self.embedder = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            device_map="auto",
            torch_dtype="auto"
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        
        self.bge_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.temp_folder = "./temp"

    def filter_text(self,text):
        text = re.sub(r"\[\'.*?\'\]", "", text)
        text = re.sub(r"<missing-text>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def _token_count(self, text):
        """Count tokens using BGE-M3's tokenizer"""
        return len(self.bge_tokenizer.encode(text, add_special_tokens=False))

    def recursive_split(self, pdf_path = None, text=None, chunk_size=400, overlap=0):
        """Traditional recursive character-based splitting with token counting"""
        if text is None:
            doc_handler = DocumentHandler()
            output_path = doc_handler.docling_serialize(pdf_path, self.temp_folder, mode=None, output_format="markdown",verbose=True,do_ocr=False,do_table_structure=False,strict_text=True,image_placeholder="\n")
            with open(output_path, "r") as f:
                text = f.read()
        text = self.filter_text(text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=self._token_count,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        return splitter.split_text(text)

    def cluster_semantic_split(self, pdf_path = None, text=None, target_chunk_size=400):
        """Cluster-based semantic chunking using BGE-M3 embeddings"""
        if text is None:
            doc_handler = DocumentHandler()
            output_path = doc_handler.docling_serialize(pdf_path, self.temp_folder, mode=None, output_format="markdown",verbose=True,do_ocr=False,do_table_structure=False,strict_text=True,image_placeholder="\n")
            with open(output_path, "r") as f:
                text = f.read()
        text = self.filter_text(text)
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            length_function=self._token_count,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        segments = base_splitter.split_text(text)
        
        embeddings = np.array(self.embedder.encode(segments)['dense_vecs'])
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            metric="cosine",
            linkage="average"
        ).fit(embeddings)
        
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            clusters.setdefault(label, []).append(segments[idx])
        
        final_chunks = []
        for cluster_segments in clusters.values():
            current_chunk = []
            current_token_count = 0
            for segment in cluster_segments:
                seg_tokens = self._token_count(segment)
                if current_token_count + seg_tokens > target_chunk_size and current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                    current_chunk = [segment]
                    current_token_count = seg_tokens
                else:
                    current_chunk.append(segment)
                    current_token_count += seg_tokens
            if current_chunk:
                final_chunks.append(" ".join(current_chunk))
                
        return final_chunks

    def _merge_cluster_segments(self, segments, target_size):
        """Optimized cluster merging with precomputed lengths"""
        current_chunk = []
        current_count = 0
        chunk_list = []
        precomputed_counts = [self._token_count(s) for s in segments]

        for seg, count in zip(segments, precomputed_counts):
            if current_count + count > target_size and current_chunk:
                chunk_list.append(" ".join(current_chunk))
                current_chunk = [seg]
                current_count = count
            else:
                current_chunk.append(seg)
                current_count += count

        if current_chunk:
            chunk_list.append(" ".join(current_chunk))
            
        return chunk_list

    def llm_semantic_split(self, pdf_path=None, text=None, target_chunk_size=400, batch_size=5):
        """Optimized LLM-assisted semantic chunking using batch processing"""
        if text is None:
            doc_handler = DocumentHandler()
            output_path = doc_handler.docling_serialize(pdf_path, self.temp_folder, mode=None, 
                                                    output_format="markdown", do_ocr=False,
                                                    strict_text=True)
            with open(output_path, "r") as f:
                text = f.read()
        text = self.filter_text(text)
        
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            length_function=self._token_count,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        segments = base_splitter.split_text(text)
        seg_token_counts = [self._token_count(seg) for seg in segments]
        
        final_chunks = []
        current_chunk = []
        current_token_count = 0
        i = 0
        n = len(segments)

        while i < n:
            while i < n and current_token_count + seg_token_counts[i] <= target_chunk_size:
                current_chunk.append(segments[i])
                current_token_count += seg_token_counts[i]
                i += 1
            
            if i >= n:
                break  # End of document
            
            batch_segments = segments[i:i+batch_size]
            batch_counts = seg_token_counts[i:i+batch_size]
            
            context = " ".join(current_chunk[-2:])  # Shorter context
            options = "\n".join([f"{n+1}. {seg[:150]}..." for n, seg in enumerate(batch_segments)])
            prompt = f"""Identify the best split point from these options:
    Current context: {context[-500:]}
    Options:
    {options}
    Respond with the option number (1-{len(batch_segments)}) or 0. Only respond with a single number."""

            # Efficient generation setup
            messages = [{"role": "user", "content": prompt}]
            text_prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(text_prompt, return_tensors="pt").to(self.llm.device)
            
            # Faster generation parameters
            outputs = self.llm.generate(
                inputs.input_ids,
                max_new_tokens=2,
                temperature=0.1,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                do_sample=False
            )
            
            # Parse response
            response = self.llm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:], 
                skip_special_tokens=True
            ).strip()
            
            try:
                split_point = int(response[0])  # Take first digit
            except:
                split_point = 0

            if 1 <= split_point <= len(batch_segments):
                # Split at chosen point
                split_idx = split_point - 1
                for j in range(split_idx + 1):
                    if current_token_count + batch_counts[j] > target_chunk_size * 1.3:
                        break  # Prevent excessive chunk size
                    current_chunk.append(batch_segments[j])
                    current_token_count += batch_counts[j]
                final_chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_token_count = 0
                i += split_idx + 1
            else:
                # Add entire batch to current chunk
                for j in range(len(batch_segments)):
                    current_chunk.append(batch_segments[j])
                    current_token_count += batch_counts[j]
                i += len(batch_segments)

        # Add final chunk
        if current_chunk:
            final_chunks.append(" ".join(current_chunk))
            
        return final_chunks

pdf_path = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/demos/cs_ai_2024_pdfs/test.pdf"
chunker = SemanticChunker()
import time

start = time.time()
recursive_chunks = chunker.recursive_split(pdf_path)
end = time.time()
output_file = "./test.txt"
i = 1
with open(output_file, "w") as f:
    for chunk in recursive_chunks:
        to_print = f"Chunk {i}:\n{chunk}\n\n"
        f.write(to_print)
        i += 1
    f.write(f"Time taken: {end-start} seconds")

start = time.time()
cluster_chunks = chunker.cluster_semantic_split(pdf_path)
end = time.time()
output_file = "./test_cluster.txt"
#open and write the cluster_chunks to the output file
i = 1
with open(output_file, "w") as f:
    for chunk in cluster_chunks:
        to_print = f"Chunk {i}:\n{chunk}\n\n"
        f.write(to_print)
        i += 1
    f.write(f"Time taken: {end-start} seconds")
"""
start = time.time()
llm_chunks = chunker.llm_semantic_split(pdf_path)
end = time.time()
output_file = "./test_llm.txt"
#open and write the llm_chunks to the output file
i = 1
with open(output_file, "w") as f:
    for chunk in llm_chunks:
        to_print = f"Chunk {i}:\n{chunk}\n\n"
        f.write(to_print)
        i += 1
    f.write(f"Time taken: {end-start} seconds")
    
"""