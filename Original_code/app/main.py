from __future__ import annotations

from fastapi import FastAPI, Depends
from fastapi.openapi.utils import get_openapi 

from app.core.logging import setup_logging
from app.api.routers.health import router as health_router
from app.api.routers.retrieval import router as retrieval_router
from app.api.routers.hybrid import router as hybrid_router
from app.api.routers.answers import router as answers_router
from app.api.routers.admin import router as admin_router
from app.api.routers.ingest import router as ingest_router
from app.api.routers.chunks import router as chunks_router
from app.api.routers.prompts import router as prompts_router

from app.middleware.auth import AuthMiddleware
from app.core.dependencies import auth_headers
 
 
def create_app() -> FastAPI:
    # Initialize logging
    setup_logging()
   
    app = FastAPI(
        title="LLM Gateway API",
        version="1.0.0",
        description="API requiring authentication headers"
    )
    
    app.add_middleware(AuthMiddleware)
    
    # Health check - no auth
    app.include_router(health_router, prefix="/api")
    
    # All other routers - show headers in docs
    app.include_router(retrieval_router, prefix="/api", dependencies=[Depends(auth_headers)])
    app.include_router(hybrid_router, prefix="/api", dependencies=[Depends(auth_headers)])
    app.include_router(answers_router, prefix="/api", dependencies=[Depends(auth_headers)])
    app.include_router(admin_router, prefix="/api", dependencies=[Depends(auth_headers)])
    app.include_router(ingest_router, prefix="/api", dependencies=[Depends(auth_headers)])
    app.include_router(chunks_router, prefix="/api", dependencies=[Depends(auth_headers)])
    app.include_router(prompts_router, prefix="/api", dependencies=[Depends(auth_headers)])
    
    return app


app = create_app()