import os
import data_loader
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np


class VectorStore:
    def __init__(self):
        self.index_path = "data/vector_index.faiss"
        self.chunks_path = "data/chunks.pkl"
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.chunks = None
        self.index = None
        self.load_index()

    def create_index(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100, add_start_index=True
        )
        documents = data_loader.fetch_content_main_page()
        self.chunks = text_splitter.split_documents(documents)
        print(f"Всего создано чанков: {len(self.chunks)}")
        print(f"Пример чанка:\n{self.chunks[0].page_content[:1000]}...")

        text_embeddings = self.embedding_model.encode([chunk.page_content for chunk in self.chunks])
        self.index = faiss.IndexFlatL2(text_embeddings.shape[1])
        self.index.add(text_embeddings)
        print("Векторное хранилище успешно создано")

        self.save_index()

    def save_index(self):
        # if self.use_gpu:
        #     index_cpu = faiss.index_gpu_to_cpu(self.index)
        #     faiss.write_index(index_cpu, self.index_path)
        # else:
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"[INFO] Index saved to {self.index_path}")
        print(f"[INFO] Chunks saved to {self.chunks_path}")

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            print("Loading existing index from disk...")
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            return
        print("[INFO] Creating new FAISS index...")
        # if self.use_gpu:
        #     res = faiss.StandardGpuResources()
        #     index = faiss.index_cpu_to_gpu(res, 0, index)
        self.create_index()

    def find_similar(self, question:str, k:int=10):
        question_embedding = np.array(self.embedding_model.encode(question))
        dists, inds = self.index.search(question_embedding.reshape(1, -1), k=k)
        retrieved_chunk = [self.chunks[i] for i in inds.tolist()[0]]
        for d, i in zip(dists[0], inds.tolist()[0]):
            print(d, self.chunks[i])
        return retrieved_chunk

if __name__ == "__main__":
    vs = VectorStore()
    vs.find_similar("Как провезти оружие?")