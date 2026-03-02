from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from langchain_openai import OpenAIEmbeddings

from app.core.config import config
logger = logging.getLogger(__name__)

@dataclass
class _TokenCache:
    access_token: Optional[str] = None
    expires_at: float = 0.0


_TOKEN_CACHE = _Token_CACHE = _TokenCache()  # backwards-friendly name

# Reuse sessions for keep-alive / connection pooling
_TOKEN_SESSION = requests.Session()
_EMBEDDINGS_SESSION = requests.Session()


def get_access_token(*, force_refresh: bool = False) -> str:
    """Fetch an Azure AD token for the LLM gateway (cached per process).

    Without caching, every `/retrieve-chunks-hybrid` request triggers an AAD token call,
    which becomes a bottleneck and causes client-side timeouts under load.
    """

    if not (config.client_id and config.client_secret and config.tenant_id):
        logger.error("Missing CLIENT_ID/CLIENT_SECRET/TENANT_ID in environment")
        raise RuntimeError("Missing CLIENT_ID/CLIENT_SECRET/TENANT_ID in environment")


    token_url = f"https://login.microsoftonline.com/{config.tenant_id}/oauth2/v2.0/token"
    token_payload = {
        "grant_type": "client_credentials",
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "scope": "api://llm-gateway.lilly.com/.default",
    }

    # Bounded timeouts: (connect timeout, read timeout)
    try:
        token_response = _TOKEN_SESSION.post(token_url, data=token_payload, timeout=(540, 300))
        token_response.raise_for_status()
        data = token_response.json()
        logger.info(f"Access token fetched successfully")
        return data["access_token"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch access token: {type(e).__name__}: {str(e)}")
        raise
 
 
def get_embeddings(texts: list[str], access_token: str):
    """Call the embeddings endpoint and return the 'data' list."""

    if not (config.base_url and config.llm_gateway_key):
        logger.error("Missing BASE_URL/LLM_GATEWAY_KEY in environment")
        raise RuntimeError("Missing BASE_URL/LLM_GATEWAY_KEY in environment")  
 
    llm_gateway_url = f"{config.base_url}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-LLM-Gateway-Key": config.llm_gateway_key,
        "Content-Type": "application/json",
    }
    payload = {"model": config.embedding_model, "input": texts}
 
    try:
        # Use json= for correctness and keep-alive pooling via Session.
        response = _EMBEDDINGS_SESSION.post(llm_gateway_url, headers=headers, json=payload, timeout=(540, 200))
        response.raise_for_status()
        logger.info(f"Embeddings fetched successfully for {len(texts)} texts")
        return response.json().get("data", [])
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to fetch embeddings: {e.response.status_code} - {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Embeddings request failed: {type(e).__name__}: {str(e)}")
        raise
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse embeddings response: {type(e).__name__}: {str(e)}")
#         raise

def get_embeddings_v2() -> OpenAIEmbeddings:
    """Create an OpenAIEmbeddings client configured for the LLM gateway."""

    logger.info("Initializing OpenAIEmbeddings client for LLM gateway")
    
    try:
        access_token = get_access_token()
        logger.info("Access token retrieved successfully")
    except Exception as e:
        logger.error(f"Failed to retrieve access token: {type(e).__name__}: {str(e)}")
        raise

    try:
        logger.debug(f"Configuring OpenAIEmbeddings with base_url={config.base_url}, model={config.embedding_model}")
        embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=access_token,
            base_url=config.base_url,
            default_headers={
                "X-LLM-Gateway-Key": config.llm_gateway_key,
            },
        )
        logger.info("OpenAIEmbeddings client created successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OpenAIEmbeddings client: {type(e).__name__}: {str(e)}")
        raise