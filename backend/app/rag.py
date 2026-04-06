from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma

class RagService:
    def __init__(self, settings):
        self.settings = settings
        self.documents = [
            Document(
                page_content="O LangChain é uma estrutura de código aberto projetada para facilitar a criação de aplicações que utilizam grandes modelos de linguagem (LLMs). Ele permite o desenvolvimento de aplicativos como chatbots e agentes virtuais, integrando diferentes componentes e funcionalidades para simplificar o processo de desenvolvimento. Além disso, o LangChain é utilizado em casos de uso como sumarização de texto e análise de código, tornando-se uma ferramenta versátil para desenvolvedores que trabalham com inteligência artificial e processamento de linguagem natural.", 
                metadata={"source": "doc1"},
            ),

            Document(
                page_content="Geração aumentada por recuperação (RAG) é uma estrutura de IA que combina duas técnicas; primeiro, ele recupera informações relevantes de fontes externas, como bancos de dados, documentos ou a Web. Uma vez reunida essa informação, ela é usada para informar e aprimorar a geração de respostas.",
                metadata={"source": "doc2"},
            ),

            Document(
                page_content="Embedding é uma técnica utilizada em aprendizado de máquina e processamento de linguagem natural que representa dados, como palavras ou imagens, em vetores de alta dimensão. Esses vetores permitem que algoritmos capturem relações semânticas, onde palavras ou dados semelhantes são mapeados para pontos próximos no espaço vetorial. Essa abordagem é fundamental para melhorar a eficiência e a precisão em tarefas como busca semântica e análise de sentimentos.",
                metadata={"source": "doc3"},
            ),
        ]

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
    
    def build_vector_store(self):
        vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.get_embeddings(),
            collection_name="rag_teste",
        )
        return vector_store

    def ask(self, question: str):
        vector_store = self.build_vector_store()
        docs = vector_store.similarity_search(question, k=2)

        context = "\n\n".join(
            [f"fonte: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
        )
        
        prompt = f"""você pe um assistente que responde a perguntas com base no contexto fornecido. 
        Se a resposta não puder ser determinada com base no contexto, responda exatamente: "Eu não encontrei essa informação nos documentos enviados". 
        
        Contexto: {context}
        Pergunta: {question}
        """

        answer = self.get_chat_model().invoke(prompt).content
        return {"answer": answer, "sources": docs}

