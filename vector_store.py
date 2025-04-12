import os
os.environ["HF_HOME"] = "C:/Games/hf/huggingface"

import data_loader
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np


class VectorStore:
    def __init__(self):
        self.index_path = "data/vector_index.faiss"
        self.chunks_path = "data/chunks.pkl"
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.chunks = None
        self.index = None
        self.documents = None
        self.load_index()

    def get_documents(self):
        return self.chunks

    def create_index(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=500
        )
        self.documents = data_loader.fetch_content_main_page()
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Всего создано чанков: {len(self.chunks)}")

        text_embeddings = self.embedding_model.encode([chunk.page_content for chunk in self.chunks])
        #faiss.normalize_L2(text_embeddings)
        self.index = faiss.IndexFlatL2(text_embeddings.shape[1])
        self.index.add(text_embeddings)
        print("Векторное хранилище успешно создано")

        self.save_index()

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"[INFO] Index saved to {self.index_path}")
        print(f"[INFO] Chunks saved to {self.chunks_path}")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            print("[INFO] Loading existing index from disk...")
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print("[INFO] Index successfully loaded.")
            return
        print("[INFO] Creating new FAISS index...")
        self.create_index()

    def find_similar(self, question:str, k:int=5):
        question_embedding = np.array(self.embedding_model.encode(question)).reshape(1, -1)
        #faiss.normalize_L2(question_embedding)
        dists, inds = self.index.search(question_embedding, k=k)
        retrieved_chunk = [self.chunks[i] for i in inds.tolist()[0]]
        for d, i in zip(dists[0], inds.tolist()[0]):
            print(d, self.chunks[i])
        return retrieved_chunk