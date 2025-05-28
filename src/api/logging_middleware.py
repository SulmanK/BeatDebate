"""
FastAPI Logging Middleware for BeatDebate

Provides comprehensive logging for all API requests and responses including:
- Request/response timing
- Status codes and error tracking
- Request IDs for tracing
- Performance monitoring
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from ..utils.logging_config import get_logger, log_api_request, set_request_context
except RuntimeError:
    # Logging not initialized yet - will use fallback
    get_logger = None
    log_api_request = None
    set_request_context = None


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests and responses.
    
    Provides:
    - Request/response timing
    - Status code tracking
    - Error logging
    - Request ID generation for tracing
    - Performance monitoring
    """
    
    def __init__(self, app, exclude_paths: list = None):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application
            exclude_paths: List of paths to exclude from logging
        """
        super().__init__(app)
        self.logger = get_logger("api.middleware") if get_logger else None
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging."""
        
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        
        # Set request context for structured logging
        if set_request_context:
            set_request_context(
                request_id=request_id,
                user_id=self._extract_user_id(request)
            )
        
        # Log request start
        start_time = time.time()
        
        if self.logger:
            self.logger.info(
                "api_request_start",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                client_ip=self._get_client_ip(request)
            )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log successful response
            if self.logger:
                self.logger.info(
                    "api_request_complete",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    duration_seconds=round(duration, 4),
                    response_size=response.headers.get("content-length", "unknown")
                )
            
            # Log to performance metrics
            if log_api_request:
                log_api_request(
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    duration=duration,
                    request_id=request_id
                )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration for error case
            duration = time.time() - start_time
            
            # Log error
            if self.logger:
                self.logger.error(
                    "api_request_error",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_seconds=round(duration, 4)
                )
            
            # Re-raise the exception
            raise
    
    def _extract_user_id(self, request: Request) -> str:
        """Extract user ID from request if available."""
        # Check for user ID in headers, query params, or session
        user_id = (
            request.headers.get("X-User-ID") or
            request.query_params.get("user_id")
        )
        
        # Try to get from session if available (only if SessionMiddleware is installed)
        try:
            if hasattr(request, "session") and "session" in request.scope:
                user_id = user_id or request.session.get("user_id")
        except (AttributeError, AssertionError):
            # SessionMiddleware not installed or session not available
            pass
        
        return user_id or "anonymous"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first (for proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Specialized middleware for performance monitoring.
    
    Tracks:
    - Slow requests (configurable threshold)
    - Memory usage
    - Database query times
    - External API call times
    """
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        """
        Initialize performance logging middleware.
        
        Args:
            app: FastAPI application
            slow_request_threshold: Time in seconds to consider a request slow
        """
        super().__init__(app)
        self.logger = get_logger("performance") if get_logger else None
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log slow requests
            if duration > self.slow_request_threshold and self.logger:
                self.logger.warning(
                    "slow_request",
                    method=request.method,
                    url=str(request.url),
                    duration_seconds=round(duration, 4),
                    threshold_seconds=self.slow_request_threshold
                )
            
            # Log performance metrics for specific endpoints
            if self._should_track_performance(request.url.path) and self.logger:
                self.logger.info(
                    "endpoint_performance",
                    endpoint=request.url.path,
                    method=request.method,
                    duration_seconds=round(duration, 4),
                    status_code=response.status_code
                )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            if self.logger:
                self.logger.error(
                    "request_exception",
                    method=request.method,
                    url=str(request.url),
                    duration_seconds=round(duration, 4),
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            
            raise
    
    def _should_track_performance(self, path: str) -> bool:
        """Determine if we should track performance for this endpoint."""
        # Track performance for key endpoints
        tracked_endpoints = [
            "/api/recommendations",
            "/api/chat",
            "/api/search"
        ]
        
        return any(tracked in path for tracked in tracked_endpoints) 