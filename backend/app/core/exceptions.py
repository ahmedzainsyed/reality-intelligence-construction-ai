"""Custom application exceptions."""
from typing import Optional, Dict, Any

class RIPBaseException(Exception):
    def __init__(self, message: str, status_code: int = 500,
                 error_code: str = "INTERNAL_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

class AuthenticationError(RIPBaseException):
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401, "AUTHENTICATION_FAILED")

class NotFoundError(RIPBaseException):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, 404, "NOT_FOUND")

class ValidationError(RIPBaseException):
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, 422, "VALIDATION_ERROR", details)

class ProcessingError(RIPBaseException):
    def __init__(self, message: str):
        super().__init__(message, 500, "PROCESSING_ERROR")
