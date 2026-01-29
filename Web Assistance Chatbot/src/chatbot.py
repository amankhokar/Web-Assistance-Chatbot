from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


class WebsiteChatbot:
    def __init__(self, vector_store, model_name="phi3", temperature=0.1):
        self.vector_store = vector_store

        self.llm = Ollama(
            model="phi3",
            temperature=temperature,
            num_ctx=2048,
            num_predict=256
        )

        self.prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the given context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

    def answer_question(self, question):
        docs = self.vector_store.similarity_search(question)

        if not docs:
            return {"answer": "I don't know", "sources": []}

        context = "\n\n".join([doc.page_content for doc in docs])

        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": response,
            "sources": docs
        }

    def clear_memory(self):
        pass
