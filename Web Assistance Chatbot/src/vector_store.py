from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, persist_directory, embedding_model):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.client = Client(
            Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name="website_documents"
        )

    def clear_collection(self):
        self.client.delete_collection("website_documents")
        self.collection = self.client.get_or_create_collection(
            name="website_documents"
        )

    def add_documents(self, documents):
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embedding_model.encode(texts).tolist()

            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=[str(i) for i in range(len(texts))]
            )
            return True
        except Exception as e:
            print("VectorStore Error:", e)
            return False

    def similarity_search(self, query, k=4):
        embedding = self.embedding_model.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append(
                type("Doc", (), {
                    "page_content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })()
            )
        return docs

    def get_collection_stats(self):
        return {
            "total_documents": self.collection.count(),
            "embedding_model": self.embedding_model_name
        }
