from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from .pdf import extract_text_from_pdf


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    # divide um texto em pedaços menores para facilitar a busca semântica
    # O overlap repete um pedaço do texto anterior para manter o contexto entre os chunks
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size    # define o fim do chunk com base no tamanho configurado
        chunks.append(text[start:end])  # recorta o trecho e adiciona à lista
        start += chunk_size - overlap   # avança, mas "volta" um pouco para criar a sobreposição
    return chunks

class RagService:
    # serviço central que orquestra todo o pipeline de RAG:
    # carrega documentos → gera embeddings → armazena vetores → busca → gera resposta

    def __init__(self, settings):
        self.settings = settings
        # o banco de vetores começa vazio — só é criado quando um PDF for enviado
        self.vector_store = None

    # cria e retorna o modelo de embeddings da Azure
    def get_embeddings(self):
        # retorna o modelo de embeddings da Azure
        return AzureOpenAIEmbeddings(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
            azure_deployment=self.settings.azure_openai_embedding_deployment,
        )

    # cria e retorna o modelo de linguagem da Azure usado para gerar a resposta final
    # temperature=0 deixa as respostas mais precisas e menos aleatórias
    def get_chat_model(self):
        # retorna o LLM da Azure para gerar a resposta final
        return AzureChatOpenAI(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
            azure_deployment=self.settings.azure_openai_chat_deployment,
            temperature=0,
        )

    def build_documents_from_pdf(self, pdf_path: str) -> list[Document]:
        # extrai o texto do PDF e divide em chunks prontos para indexação
        text = extract_text_from_pdf(Path(pdf_path))
        chunks = split_text(text)

        documents = []
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": Path(pdf_path).name,
                        "chunk_id": f"chunk_{index}",
                    },
                )
            )
        return documents

    def build_vector_store(self, pdf_path: str):
        # constrói o banco de vetores e salva em self.vector_store para uso no ask()
        documents = self.build_documents_from_pdf(pdf_path)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.get_embeddings(),    # função que gera os vetores para cada chunk
            collection_name="rag_pdf",
        )

    def ask(self, pdf_path: str, question: str):
        # constrói o vector store com o PDF recebido antes de fazer qualquer busca
        self.build_vector_store(pdf_path)

        # busca os k=2 chunks mais semanticamente próximos da pergunta usando similaridade de cosseno
        docs = self.vector_store.similarity_search(question, k=3)

        # monta o contexto que será injetado no prompt, incluindo a fonte e o conteúdo de cada chunk
        context = "\n\n".join(
            [
                f"fonte: {doc.metadata['source']} | chunk: {doc.metadata['chunk_id']}\n{doc.page_content}" 
                for doc in docs
            ]
        )

        prompt = f"""você é um assistente que responde a perguntas com base no contexto fornecido. 
        Se a resposta não puder ser determinada com base no contexto, responda exatamente: "Eu não encontrei essa informação nos documentos enviados". 
        
        Contexto: {context}
        Pergunta: {question}
        """

        # envia o prompt para o LLM e extrai o texto da resposta
        answer = self.get_chat_model().invoke(prompt).content
        return {"answer": answer, "sources": docs}

