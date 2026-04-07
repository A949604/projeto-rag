from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

def split_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    # divide um texto em pedaços menores
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

    # cria e retorna o modelo de embeddings da Azure
    def get_embeddings(self):
        embeddings = AzureOpenAIEmbeddings(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
            azure_deployment=self.settings.azure_openai_embedding_deployment,
        )
        return embeddings

    # cria e retorna o modelo de linguagem da Azure usado para gerar a resposta final
    # temperature=0 deixa as respostas mais precisas e menos aleatórias
    def get_chat_model(self):
        model = AzureChatOpenAI(
            api_key=self.settings.azure_openai_api_key,
            azure_endpoint=self.settings.azure_openai_endpoint,
            api_version=self.settings.azure_openai_api_version,
            azure_deployment=self.settings.azure_openai_chat_deployment,
            temperature=0,
        )
        return model

    # simula uma base de conhecimento com documentos estáticos
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
            # quebra o texto de cada documento em chunks menores antes de indexar
            chunks = split_text(item["text"])
            for index, chunk in enumerate(chunks):
                # cada chunk vira um Document independente com metadados de rastreabilidade
                # o metadata permite saber depois de qual documento e qual pedaço veio a resposta
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": item["source"],   # identificador do doc original
                            "chunk_id": f"{item['source']}_{index}",    # identificador único do chunk
                        },
                    )
                )
        return documents

    def build_vector_store(self):
        # constrói o banco de vetores a partir dos documentos carregados
        # o Chroma converte cada chunk em embedding e armazena tudo em memória
        documents = self.load_documents()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.get_embeddings(),    # função que gera os vetores para cada chunk
            collection_name="rag_teste_chunks",
        )
        return vector_store

    def ask(self, question: str):
        # método recebe uma pergunta e retorna a resposta gerada com base nos documentos

        # reconstrói o vector store a cada chamada
        vector_store = self.build_vector_store()

        # busca os k=2 chunks mais semanticamente próximos da pergunta usando similaridade de cosseno
        docs = vector_store.similarity_search(question, k=2)

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

