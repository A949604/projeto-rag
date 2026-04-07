from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

def split_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

class RagService:
    def __init__(self, settings):
        self.settings = settings

    def get_embeddings(self):
        embeddings = AzureOpenAIEmbeddings(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
            azure_deployment=self.settings.azure_openai_embedding_deployment,
        )
        return embeddings

    def get_chat_model(self):
        model = AzureChatOpenAI(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
            azure_deployment=self.settings.azure_openai_chat_deployment,
            temperature=0,
        )
        return model

    def load_documents(self):
        docs = [
            {
                "source": "doc1",
                "text": (
                    "RAG, ou Retrieval-Augmented Generation, é uma técnica que combina a geração de texto com a recuperação de informações relevantes. "
                ),
            },
            {
                "source": "doc2",
                "text": (
                    "O processo de RAG envolve a recuperação de documentos relevantes com base em uma consulta, seguida pela geração de uma resposta que incorpora as informações recuperadas. "
                ),
            },
        ]

        documents = []

        for item in docs:
            chunks = split_text(item["text"])
            for index, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": item["source"],
                            "chunk_id": f"{item['source']}_{index}",
                        },
                    )
                )
        return documents
    
    def build_vector_store(self):
        documents = self.load_documents()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.get_embeddings(),
            collection_name="rag_teste_chunks",
        )
        return vector_store

    def ask(self, question: str):
        vector_store = self.build_vector_store()
        docs = vector_store.similarity_search(question, k=2)

        context = "\n\n".join(
            [
                f"fonte: {doc.metadata['source']} | chunk: {doc.metadata['chunk_id']}\n{doc.page_content}" 
                for doc in docs
            ]
        )
        
        prompt = f"""você pe um assistente que responde a perguntas com base no contexto fornecido. 
        Se a resposta não puder ser determinada com base no contexto, responda exatamente: "Eu não encontrei essa informação nos documentos enviados". 
        
        Contexto: {context}
        Pergunta: {question}
        """

        answer = self.get_chat_model().invoke(prompt).content
        return {"answer": answer, "sources": docs}

