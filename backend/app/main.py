from fastapi import FastAPI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel#, Field

class Settings(BaseSettings):
    # azure_openai_api_key: str = Field(default="", env="AZURE_OPENAI_API_KEY")
    # azure_openai_endpoint: str = Field(default="", env="AZURE_OPENAI_ENDPOINT")
    # azure_openai_api_version: str = Field(default="", env="AZURE_OPENAI_API_VERSION")
    # azure_openai_chat_deployment: str = Field(default="", env="AZURE_OPENAI_CHAT_DEPLOYMENT")

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = ""
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
app = FastAPI()

@app.get("/")
def home():
    return {"message": "funcionando!"}

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