import os
from typing import List, cast
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

load_dotenv()

class Config(BaseSettings):

    # AZURE_OPENAI_API_KEY: str = cast(str, os.getenv("AZURE_OPENAI_API_KEY"))
    # AZURE_OPENAI_ENDPOINT: str = cast(str, os.getenv("AZURE_OPENAI_ENDPOINT"))
    client_id:str=Field(alias="LLM_GATEWAY_CLIENT_ID")
    
    client_secret:str=Field(alias="LLM_GATEWAY_CLIENT_SECRET")
    tenant_id:str=Field(alias="LLM_GATEWAY_TENANT_ID")
    llm_gateway_key:str=Field(alias="LLM_GATEWAY_KEY")
    base_url:str=Field(alias="LLM_GATEWAY_BASE_URL")
    embedding_model:str=Field(alias="EMBEDDING_MODEL", default="text-embedding-3-large")
    
    postgres_user:str=cast(str, os.getenv("DB_USER"))
    postgres_password:str=cast(str, os.getenv("DB_PASS"))
    postgres_host:str=cast(str, os.getenv("DB_HOST"))
    postgres_port:str=cast(str, os.getenv("DB_PORT"))
    postgres_db:str=cast(str, os.getenv("DB_NAME"))
    
    ibu_client_id:str=Field(alias="CLIENT_ID")
    ibu_client_secret:str=Field(alias="CLIENT_SECRET")
 
    @property
    def RDS_URI(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
 
    model_config = {"case_sensitive": False}


def get_allowed_origins(config: Config) -> List[str]:
    origins = ["*"]
    # if config.ALLOWED_ORIGINS_REGISTRY:
    #     with open(config.ALLOWED_ORIGINS_REGISTRY, "r") as origins_file:
    #         origins = []
    #         for line in origins_file:
    #             origins.append(line.strip())
    return origins


config = Config()
print(f"CLIENT_ID: {config.client_id}")