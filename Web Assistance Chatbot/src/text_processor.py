from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, text, metadata=None):
        chunks = self.splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]
