"""
Middleware package for the application.

Contains:
- AuthMiddleware: Validates client authentication headers
"""

from app.middleware.auth import AuthMiddleware

__all__ = ["AuthMiddleware"]