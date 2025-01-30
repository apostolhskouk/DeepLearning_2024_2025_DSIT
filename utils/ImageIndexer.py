import os
import numpy as np
import faiss
from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoImageProcessor, ResNetModel
from torch.nn import AdaptiveAvgPool2d
from pathlib import Path
from byaldi import RAGMultiModalModel
from torch.utils.data import DataLoader
from colpali_engine.models import ColPali, ColQwen2Processor
from byaldi.objects import Result
import base64
import io
from tqdm import tqdm

class ResNetIndexer:
    @staticmethod
    def create_index(input_folder, index_path, model_name="microsoft/resnet-50", batch_size=16):
        os.makedirs(index_path, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ResNetModel.from_pretrained(model_name).to(device)
        model.eval()
        pooler = AdaptiveAvgPool2d((1, 1))
        
        image_paths = []
        embeddings = []
        
        for fname in os.listdir(input_folder):
            if fname.endswith(".png"):
                path = os.path.join(input_folder, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = pooler(outputs.last_hidden_state).squeeze()
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                    
                    embeddings.append(embedding.cpu().numpy())
                    image_paths.append(path)
                except Exception as e:
                    print(f"Skipped {fname}: {str(e)}")
        
        embeddings_np = np.array(embeddings).astype("float32")
        np.save(os.path.join(index_path, "embeddings.npy"), embeddings_np)
        np.save(os.path.join(index_path, "image_paths.npy"), np.array(image_paths))
        
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)
        faiss.write_index(index, os.path.join(index_path, "faiss_index.bin"))

    @staticmethod
    def query_by_image(query_path, index_path, top_k=5):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        model.eval()
        pooler = AdaptiveAvgPool2d((1, 1))
        
        index = faiss.read_index(os.path.join(index_path, "faiss_index.bin"))
        image_paths = np.load(os.path.join(index_path, "image_paths.npy"))
        
        img = Image.open(query_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        query_emb = pooler(outputs.last_hidden_state).squeeze()
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0).cpu().numpy().astype("float32")
        
        distances, indices = index.search(query_emb.reshape(1, -1), top_k)
        return [(image_paths[i], d) for i, d in zip(indices[0], distances[0])]

class CLIPIndexer:
    @staticmethod
    def create_index(input_folder, index_path, model_name="jinaai/jina-clip-v2", truncate_dim=1024, batch_size=4):
        os.makedirs(index_path, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        
        image_paths = []
        embeddings = []
        
        paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".png")]
        
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_imgs = []
            
            for path in batch_paths:
                try:
                    batch_imgs.append(Image.open(path).convert("RGB"))
                    image_paths.append(path)
                except Exception as e:
                    print(f"Skipped {path}: {str(e)}")
            
            if batch_imgs:
                with torch.no_grad():
                    emb = model.encode_image(batch_imgs, truncate_dim=truncate_dim)
                embeddings.append(emb)
        
        embeddings_np = np.concatenate(embeddings).astype("float32")
        np.save(os.path.join(index_path, "embeddings.npy"), embeddings_np)
        np.save(os.path.join(index_path, "image_paths.npy"), np.array(image_paths))
        
        index = faiss.IndexFlatIP(embeddings_np.shape[1])
        index.add(embeddings_np)
        faiss.write_index(index, os.path.join(index_path, "faiss_index.bin"))

    @staticmethod
    def query_by_image(query_path, index_path, top_k=5):
        return CLIPIndexer._query(query_path, index_path, top_k, "image")

    @staticmethod
    def query_by_text(query_text, index_path, top_k=5):
        return CLIPIndexer._query(query_text, index_path, top_k, "text")

    @staticmethod
    def _query(query_input, index_path, top_k, input_type):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
        model.eval()
        
        index = faiss.read_index(os.path.join(index_path, "faiss_index.bin"))
        image_paths = np.load(os.path.join(index_path, "image_paths.npy"))
        
        if input_type == "image":
            img = Image.open(query_input).convert("RGB")
            with torch.no_grad():
                query_emb = model.encode_image([img])
        else:
            with torch.no_grad():
                query_emb = model.encode_text([query_input], task="retrieval.query")
        
        distances, indices = index.search(query_emb.astype("float32"), top_k)
        return [(image_paths[i], d) for i, d in zip(indices[0], distances[0])]

from io import BytesIO

def pil_image_to_bytes(img):
    # Save the image to a BytesIO buffer
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

class ImageRAG:
    def __init__(self, model, processor, image_paths, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self.image_paths = image_paths
        self.embeddings = []
        self.image_base64s = []

    def _image_to_base64(self, image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _base64_to_image(self, base64_str):
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))

    def add_images_and_embeddings(self, images, embeddings):
        assert len(images) == len(embeddings)
        self.embeddings = embeddings
        for img in images:
            self.image_base64s.append(self._image_to_base64(img))

    def find_similar_images(self, query_image, top_k=5, batch_size=4):
        # Process the query image
        processed_query = self.processor.process_images([query_image])
        processed_query = {k: v.to(self.device) for k, v in processed_query.items()}
        
        with torch.no_grad():
            query_embedding = self.model(**processed_query)
            if self.device == "cuda":
                query_embedding = query_embedding.float()  # Ensure query_embedding is float32
            query_embedding = query_embedding.cpu()

        # Cast the query embedding to bfloat16 to match indexed embeddings' dtype
        query_embedding = query_embedding.to(torch.bfloat16)

        # Score similarity between query and indexed images
        scores = []
        for i in range(0, len(self.embeddings), batch_size):
            batch_embeddings = self.embeddings[i:i + batch_size]
            batch_scores = []
            for emb in batch_embeddings:
                similarity = torch.matmul(query_embedding, emb.t())
                max_similarity = similarity.max(dim=-1)[0]
                score = max_similarity.sum()
                batch_scores.append(score)
            scores.extend(batch_scores)

        scores = torch.tensor(scores)
        top_k_values, top_k_indices = torch.topk(scores, min(top_k, len(scores)))

        results = []
        for similarity, idx in zip(top_k_values, top_k_indices):
            results.append({
                'similarity': similarity.item(),
                'image_path': self.image_paths[idx],
                'index': idx.item()
            })
        return results
    
def get_batch_image_embeddings(images, model, processor, batch_size=2):
    """
    Generate embeddings for a batch of images.
    """
    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x)
    )
    embeddings = []
    for batch in tqdm(dataloader, desc="Processing images"):
        with torch.no_grad():
            batch = {k: v.to(model.device) for k, v in batch.items()}
            batch_embeddings = model(**batch)
            if model.device == "cuda":
                batch_embeddings = batch_embeddings.float()
            embeddings.extend(list(torch.unbind(batch_embeddings.cpu())))
    return embeddings

class ByaldiIndexer:
    @staticmethod
    def create_index(input_folder, index_path, index_name="pdfs_images", 
                    model_name="vidore/colqwen2-v1.0"):
        os.makedirs(index_path, exist_ok=True)
        
        model = RAGMultiModalModel.from_pretrained(model_name, index_root=index_path)
        metadata = [{"filename": f} for f in os.listdir(input_folder) if f.endswith(".png")]
        
        model.index(
            input_path=Path(input_folder),
            index_name=index_name,
            store_collection_with_index=False,
            metadata=metadata,
            overwrite=True
        )

    @staticmethod
    def query_by_text(query_text, index_path, index_name="pdfs_images", top_k=5):
        model = RAGMultiModalModel.from_index(index_path=index_name, index_root=index_path)
        results = model.search(query_text, k=top_k)
        return [(r.metadata['filename'], r.score) for r in results]

    @staticmethod
    def query_by_image(query_img_path, image_dir, index_path, index_name="pdfs_images", model_name="vidore/colqwen2-v1.0", top_k=5):
        model = RAGMultiModalModel.from_index(index_path=index_name, index_root=index_path)
        colpali_model = model.model.model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = ColQwen2Processor.from_pretrained(model_name)
        image_paths = [
            os.path.join(image_dir, file) for file in os.listdir(image_dir)
            if file.endswith(".png")
        ]
        extracted_images = [
            Image.open(os.path.join(image_dir, file)) for file in os.listdir(image_dir)
            if file.endswith(".png")
        ]
        image_rag = ImageRAG(colpali_model, processor, image_paths, device=device)
        # Generate embeddings for the extracted images
        embeddings = get_batch_image_embeddings(extracted_images, colpali_model, processor, batch_size=2)
        # Add images and embeddings to ImageRAG
        image_rag.add_images_and_embeddings(extracted_images, embeddings)
        results = image_rag.find_similar_images(Image.open(query_img_path), top_k=top_k)
        # Convert results to the desired format with np.str_ and np.float32
        formatted_results = [
            (np.str_(result['image_path']), np.float32(result['similarity'])) for result in results
        ]
        
        return formatted_results
    
def visualize_results(results, input_folder=None):
    plt.figure(figsize=(20, 10))
    for i, (path, score) in enumerate(results):
        try:
            # Determine the image path based on input_folder
            if input_folder is not None:
                filename = os.path.basename(path)
                img_path = os.path.join(input_folder, filename)
            else:
                img_path = path

            # Create subplot and display image
            plt.subplot(1, len(results), i+1)
            plt.imshow(Image.open(img_path))
            plt.title(f"Rank {i+1}\nScore: {score:.3f}")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
    plt.tight_layout()
    plt.show()

byaldi_folder = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/demos/index_byaldi"
clip_folder = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/demos/clip_embeddings"
resnet_folder = "/home/tolis/Desktop/tolis/DNN/project/DeepLearning_2024_2025_DSIT/demos/embeddings"