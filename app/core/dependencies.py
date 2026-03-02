from fastapi import Header
from typing import Annotated, Optional


async def auth_headers(
    x_ibu_fahad_client_id: Annotated[
        Optional[str], 
        Header(alias="X-IBU-Fahad-Client-Id", description="Client ID for API authentication")
    ] = None,
    x_ibu_fahad_client_secret: Annotated[
        Optional[str], 
        Header(alias="X-IBU-Fahad-Client-Secret", description="Client secret for API authentication")
    ] = None,
    x_ibu_fahad_client_source: Annotated[
        Optional[str], 
        Header(alias="X-IBU-Fahad-Client-Source", description="Source system (ibu-fahad or ibu-digital-person)")
    ] = None,
):
    """
    Documents required authentication headers in Swagger UI.
    
    Note: Actual validation is performed by AuthMiddleware.
    This dependency is only for OpenAPI documentation.
    """
    pass  # Don't return anything - middleware already validated