from fastapi import FastAPI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from .rag import RagService
from pathlib import Path
from .pdf import extract_text_from_pdf

# classe que carrega as configurações sensíveis a partir do arquivo .env
# isso evita deixar dados sigilosos diretamente no código
class Settings(BaseSettings):
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = ""
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    # configuração do pydantic para ler automaticamente as variáveis do arquivo .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore", # ignora variáveis do .env que não estão definidas aqui
    )

# modelo que define o formato esperado no corpo da requisição POST /ask
# o pydantic valida automaticamente se o campo "question" foi enviado e é uma string
class QuestionRequest(BaseModel):
    question: str

settings = Settings()
app = FastAPI()
rag_service = RagService(settings)

@app.get("/")
def home():
    return {"message": "funcionando!"}


# rota de teste direto com o modelo de chat, faz uma pergunta fixa sem RAG
# pra verificar se a conexão com o Azure OpenAI tá funcionando
@app.get("/chat")
def chat():
    model = AzureChatOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_chat_deployment,
        temperature=0,
    )

    response = model.invoke("O que é RAG?")
    return {"answer": response.content}


# rota de teste para o modelo de embeddings, gera o vetor de uma frase fixa
@app.get("/embedding")
def embedding():
    embeddings = AzureOpenAIEmbeddings(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
        azure_deployment=settings.azure_openai_embedding_deployment,
    )

    vector = embeddings.embed_query("testando embedding")
    return {
        "message": "embedding funcionando",
        "dimensions": len(vector),
        "preview": vector[:5],
    } 

# rota principal do sistema RAG, recebe uma pergunta e retorna a resposta com as fontes usadas
@app.post("/ask")
def ask_question(data: QuestionRequest):

    # delega a lógica de busca e resposta para o RagService
    result = rag_service.ask("./data/uploads/teste.pdf", data.question)
    return {
        "question": data.question,
        "answer": result["answer"],
        "sources": [
            {
                "source": doc.metadata.get("source"),
                "content": doc.page_content,
            }
            for doc in result["sources"]
        ],
    }


# rota de teste para extração de texto de PDF, lê um arquivo fixo e retorna um preview
@app.get("/test-pdf")
def test_pdf():
    pdf_path = Path("./data/uploads/teste.pdf")
    text = extract_text_from_pdf(pdf_path)
    return {"preview": text[:500]}