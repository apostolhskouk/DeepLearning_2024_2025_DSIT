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
    def __init__(self, index_path, model_name="microsoft/resnet-50"):
        self.index_path = index_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lazy-loaded components
        self._model = None
        self._processor = None
        self._faiss_index = None
        self._image_paths = None
        self._pooler = AdaptiveAvgPool2d((1, 1))

    def create_index(self, input_folder, batch_size=16):
        """Create and save FAISS index from images"""
        os.makedirs(self.index_path, exist_ok=True)
        self._load_model()
        
        image_paths = []
        embeddings = []
        
        for fname in tqdm(os.listdir(input_folder), desc="Indexing images"):
            if fname.endswith(".png"):
                path = os.path.join(input_folder, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    inputs = self._processor(images=img, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                    embedding = self._pooler(outputs.last_hidden_state).squeeze()
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                    
                    embeddings.append(embedding.cpu().numpy())
                    image_paths.append(path)
                except Exception as e:
                    print(f"Skipped {fname}: {str(e)}")
        
        embeddings_np = np.array(embeddings).astype("float32")
        self._save_index(embeddings_np, image_paths)
        
        # Reset cached components
        self._faiss_index = None
        self._image_paths = None

    def query_by_image(self, query_path, top_k=5):
        """Query index using an image"""
        self._load_model()
        self._load_index()
        
        img = Image.open(query_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        query_emb = self._pooler(outputs.last_hidden_state).squeeze()
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0).cpu().numpy().astype("float32")
        
        distances, indices = self._faiss_index.search(query_emb.reshape(1, -1), top_k)
        return [(self._image_paths[i], d) for i, d in zip(indices[0], distances[0])]

    def _load_model(self):
        """Lazy-load model and processor"""
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = ResNetModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def _load_index(self):
        """Lazy-load FAISS index and paths"""
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(os.path.join(self.index_path, "faiss_index.bin"))
            self._image_paths = np.load(os.path.join(self.index_path, "image_paths.npy"))

    def _save_index(self, embeddings, image_paths):
        """Save index components to disk"""
        np.save(os.path.join(self.index_path, "embeddings.npy"), embeddings)
        np.save(os.path.join(self.index_path, "image_paths.npy"), np.array(image_paths))
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(self.index_path, "faiss_index.bin"))

class CLIPIndexer:
    def __init__(self, index_path, model_name="jinaai/jina-clip-v2", truncate_dim=1024):
        self.index_path = index_path
        self.model_name = model_name
        self.truncate_dim = truncate_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lazy-loaded components
        self._model = None
        self._faiss_index = None
        self._image_paths = None

    def create_index(self, input_folder, batch_size=4):
        """Create and save FAISS index from images"""
        os.makedirs(self.index_path, exist_ok=True)
        self._load_model()
        
        image_paths = []
        embeddings = []
        paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".png")]
        
        for i in tqdm(range(0, len(paths), batch_size), desc="Indexing images"):
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
                    # Jina CLIP returns numpy arrays directly
                    emb = self._model.encode_image(batch_imgs, truncate_dim=self.truncate_dim)
                embeddings.append(emb)  # Remove .cpu().numpy()
        
        embeddings_np = np.concatenate(embeddings).astype("float32")
        self._save_index(embeddings_np, image_paths)
        
        # Reset cached components
        self._faiss_index = None
        self._image_paths = None

    def query_by_image(self, query_path, top_k=5):
        """Query index using an image"""
        return self._query(query_path, top_k, "image")

    def query_by_text(self, query_text, top_k=5):
        """Query index using text"""
        return self._query(query_text, top_k, "text")

    def _query(self, query_input, top_k, input_type):
        self._load_model()
        self._load_index()
        
        if input_type == "image":
            img = Image.open(query_input).convert("RGB")
            with torch.no_grad():
                query_emb = self._model.encode_image([img], truncate_dim=self.truncate_dim)
        else:
            with torch.no_grad():
                query_emb = self._model.encode_text([query_input], task="retrieval.query")
        
        distances, indices = self._faiss_index.search(query_emb.astype("float32"), top_k)
        return [(self._image_paths[i], d) for i, d in zip(indices[0], distances[0])]

    def _load_model(self):
        """Lazy-load model"""
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self._model.eval()

    def _load_index(self):
        """Lazy-load FAISS index and paths"""
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(os.path.join(self.index_path, "faiss_index.bin"))
            self._image_paths = np.load(os.path.join(self.index_path, "image_paths.npy"))

    def _save_index(self, embeddings, image_paths):
        """Save index components to disk"""
        np.save(os.path.join(self.index_path, "embeddings.npy"), embeddings)
        np.save(os.path.join(self.index_path, "image_paths.npy"), np.array(image_paths))
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(self.index_path, "faiss_index.bin"))


class ColpaliIndexer:
    def __init__(self, index_path, index_name="pdfs_images", model_name="vidore/colqwen2-v1.0"):
        self.index_path = Path(index_path)
        self.index_name = index_name
        self.model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Lazy-loaded components
        self._model = None
        self._processor = None
        self._faiss_index = None
        self._image_paths = None

    def create_index(self, input_folder):
        """Create and save both multimodal and FAISS indices"""
        self._ensure_model_initialized()
        self._create_multimodal_index(input_folder)
        self._create_faiss_image_index(input_folder)

    def query_by_text(self, query_text, top_k=5):
        """Query using text input"""
        self._load_multimodal_components()
        results = self._model.search(query_text, k=top_k)
        return [(r.metadata['filename'], r.score) for r in results]

    def query_by_image(self, query_img_path, top_k=5):
        """Query using image input"""
        self._load_faiss_components()
        query_embedding = self._get_query_embedding(query_img_path)
        distances, indices = self._faiss_index.search(query_embedding, top_k)
        return [(self._image_paths[i], d) for i, d in zip(indices[0], distances[0])]

    def _ensure_model_initialized(self):
        """Initialize core model components once"""
        if self._model is None:
            self._model = RAGMultiModalModel.from_pretrained(
                self.model_name, 
                index_root=self.index_path
            )
            self._processor = ColQwen2Processor.from_pretrained(self.model_name)
            self._model.model.model.to(self._device)

    def _create_multimodal_index(self, input_folder):
        """Create the multimodal RAG index"""
        metadata = [{
            "filename": os.path.abspath(os.path.join(input_folder, f))
        } for f in os.listdir(input_folder) if f.endswith(".png")]
        torch.cuda.empty_cache()
        self._model.index(
            input_path=Path(input_folder),
            index_name=self.index_name,
            store_collection_with_index=False,
            metadata=metadata,
            overwrite=True
        )
        torch.cuda.empty_cache()

    def _create_faiss_image_index(self, input_folder):
        """Create FAISS index for image embeddings"""
        image_paths, images = self._load_images(input_folder)
        embeddings = self._generate_embeddings(images)
        self._save_faiss_index(embeddings, image_paths)

    def _load_images(self, input_folder):
        """Load and validate images from directory"""
        image_paths = []
        images = []
        valid_extensions = {".png", ".jpg", ".jpeg"}
        
        for f in os.listdir(input_folder):
            file_ext = os.path.splitext(f)[1].lower()
            if file_ext in valid_extensions:
                path = os.path.join(input_folder, f)
                try:
                    images.append(Image.open(path).convert("RGB"))
                    image_paths.append(path)
                except Exception as e:
                    print(f"Skipped {path}: {str(e)}")
        return image_paths, images

    def _generate_embeddings(self, images, batch_size=1):
        """Generate image embeddings using the model"""
        dataloader = DataLoader(
            images,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: self._processor.process_images(x)
        )
        
        embeddings = []
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            with torch.no_grad():
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs = self._model.model.model(**batch)
                
                # Modified: Handle different output formats
                if isinstance(outputs, torch.Tensor):
                    emb_tensor = outputs
                elif hasattr(outputs, 'last_hidden_state'):
                    emb_tensor = outputs.last_hidden_state
                else:
                    raise ValueError("Unsupported model output format")
                
                # Handle sequence dimension if present
                if emb_tensor.dim() == 3:
                    emb_tensor = emb_tensor.mean(dim=1)  # Average pooling
                    
                batch_emb = emb_tensor.float().cpu().numpy().astype("float32")
                
                if embeddings and batch_emb.shape[1] != embeddings[0].shape[1]:
                    raise ValueError("Embedding dimension mismatch")
                
                embeddings.append(batch_emb)
                del batch, outputs
                torch.cuda.empty_cache()
                
        return np.concatenate(embeddings)
    def _save_faiss_index(self, embeddings, image_paths):
        """Save FAISS index and image paths"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        faiss.write_index(index, os.path.join(str(self.index_path), "image_index.faiss"))

        np.save(self.index_path/"image_paths.npy", np.array(image_paths))

    def _load_multimodal_components(self):
        """Lazy-load RAG model components"""
        if self._model is None:
            self._model = RAGMultiModalModel.from_index(
                index_path=self.index_name,
                index_root=self.index_path
            )
            self._model.model.model.to(self._device)

    def _load_faiss_components(self):
        """Lazy-load FAISS index components"""
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(os.path.join(str(self.index_path), "image_index.faiss"))
            self._image_paths = np.load(self.index_path/"image_paths.npy")

    def _get_query_embedding(self, query_img_path):
        """Generate embedding for query image"""
        # Ensure model AND processor are initialized
        self._ensure_model_initialized()  # <-- Add this line
        
        query_img = Image.open(query_img_path).convert("RGB")
        processed = self._processor.process_images([query_img])
        processed = {k: v.to(self._device) for k, v in processed.items()}

        with torch.no_grad():
            output = self._model.model.model(**processed)
            
            # Handle different output formats
            if isinstance(output, torch.Tensor):
                emb_tensor = output
            elif hasattr(output, 'last_hidden_state'):
                emb_tensor = output.last_hidden_state
            else:
                raise ValueError("Unsupported model output format")
            
            # Handle sequence dimension
            if emb_tensor.dim() == 3:
                emb_tensor = emb_tensor.mean(dim=1)
                
            return emb_tensor.float().cpu().numpy().astype("float32")

    
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