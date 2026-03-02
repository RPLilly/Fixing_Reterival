from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.core.config import config


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate authentication headers on all requests."""
    
    # Paths that don't require authentication
    SKIP_AUTH_PATHS = {
        "/api/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        if request.url.path in self.SKIP_AUTH_PATHS:
            return await call_next(request)
        
        # Get authentication headers
        ibu_client_id = request.headers.get("X-IBU-Fahad-Client-Id")
        ibu_client_secret = request.headers.get("X-IBU-Fahad-Client-Secret")
        ibu_client_source = request.headers.get("X-IBU-Fahad-Client-Source")
        
        # Check if all required headers are present
        if not ibu_client_id or not ibu_client_secret or not ibu_client_source:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "detail": "Missing required authentication headers: X-IBU-Fahad-Client-Id, X-IBU-Fahad-Client-Secret, X-IBU-Fahad-Client-Source"
                }
            )
        
        # Check if config values are loaded
        if not config.ibu_client_id or not config.ibu_client_secret:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Server authentication configuration is missing"
                }
            )
        
        # Validate client ID
        if ibu_client_id != config.ibu_client_id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid Auth details"}
            )
        
        # Validate client secret
        if ibu_client_secret != config.ibu_client_secret:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid Auth details"}
            )
        
        # Optional: Validate source against whitelist
        allowed_sources = ["ibu-digital-person"]
        if ibu_client_source not in allowed_sources:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": f"Invalid source"
                }
            )
        
        # Store auth info in request state for access in route handlers
        # request.state.client_id = client_id
        # request.state.source = source
        
        # Continue processing the request
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Internal server error: {str(e)}"}
            )