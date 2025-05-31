"""
BeatDebate Logging Configuration

Comprehensive logging system for the BeatDebate application that provides:
- Structured logging with JSON output
- File rotation and organization
- Component-specific log levels
- Performance monitoring
- Error tracking
- API request/response logging
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars


class BeatDebateLogger:
    """
    Centralized logging configuration for BeatDebate.
    
    Provides structured logging with:
    - File rotation by size and time
    - Component-specific log levels
    - JSON formatting for structured data
    - Console output for development
    - Separate files for different log types
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        enable_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory to store log files
            log_level: Default log level
            enable_console: Whether to enable console logging
            max_file_size: Maximum size per log file before rotation
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Component-specific log levels
        self.component_levels = {
            "main": logging.INFO,        # Main application flow
            "ui": logging.WARNING,       # Keep UI less verbose
            "external": logging.WARNING, # External API calls (less noise)
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure the complete logging system."""
        # Clear any existing configuration
        logging.getLogger().handlers.clear()
        
        # Configure structlog
        self._configure_structlog()
        
        # Setup file handlers
        self._setup_file_handlers()
        
        # Setup console handler
        if self.enable_console:
            self._setup_console_handler()
        
        # Configure component-specific loggers
        self._configure_component_loggers()
        
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
    
    def _configure_structlog(self):
        """Configure structlog for structured logging."""
        # Shared processors for all outputs
        shared_processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Configure structlog
        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _setup_file_handlers(self):
        """Setup rotating file handlers for essential log categories."""
        
        # Main application log - captures everything
        main_handler = self._create_rotating_file_handler(
            filename="beatdebate.log",
            level=self.log_level
        )
        
        # Errors only - for quick debugging
        error_handler = self._create_rotating_file_handler(
            filename="errors.log",
            level=logging.ERROR
        )
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        for handler in [main_handler, error_handler]:
            root_logger.addHandler(handler)
    
    def _create_rotating_file_handler(
        self, 
        filename: str, 
        level: int
    ) -> logging.handlers.RotatingFileHandler:
        """Create a rotating file handler with JSON formatting."""
        
        filepath = self.log_dir / filename
        
        handler = logging.handlers.RotatingFileHandler(
            filename=filepath,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        
        handler.setLevel(level)
        
        # JSON formatter for structured data
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
        )
        
        handler.setFormatter(formatter)
        return handler
    
    def _setup_console_handler(self):
        """Setup console handler for development."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Colored console output for development
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )
        
        console_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(console_handler)
    
    def _configure_component_loggers(self):
        """Configure specific loggers for different components."""
        
        # If global log level is DEBUG, don't override component levels
        # This allows all modules to show debug logs when LOG_LEVEL=DEBUG
        if self.log_level == logging.DEBUG:
            # When debugging, we want to see logs from all components
            # Only reduce noise from truly external libraries
            external_modules = ["httpx", "requests", "urllib3", "urllib3.connectionpool"]
            for ext_module in external_modules:
                logger = logging.getLogger(ext_module)
                logger.setLevel(logging.WARNING)  # Still reduce noise from HTTP libraries
            return
        
        # Normal operation: apply component-specific log levels
        # UI loggers (less verbose)
        ui_modules = ["src.ui.chat_interface", "src.ui.response_formatter"]
        for ui_module in ui_modules:
            logger = logging.getLogger(ui_module)
            logger.setLevel(self.component_levels["ui"])
        
        # External API loggers (reduce noise)
        external_modules = ["httpx", "requests", "urllib3", "urllib3.connectionpool"]
        for ext_module in external_modules:
            logger = logging.getLogger(ext_module)
            logger.setLevel(self.component_levels["external"])
    
    def get_logger(self, name: str) -> structlog.BoundLogger:
        """Get a structured logger for a specific component."""
        return structlog.get_logger(name)
    
    def set_request_context(self, request_id: str, user_id: Optional[str] = None):
        """Set context for the current request."""
        clear_contextvars()
        bind_contextvars(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def log_performance(
        self, 
        operation: str, 
        duration: float, 
        **kwargs
    ):
        """Log performance metrics."""
        perf_logger = self.get_logger("performance")
        perf_logger.info(
            "performance_metric",
            operation=operation,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_api_request(
        self,
        method: str,
        url: str,
        status_code: int,
        duration: float,
        **kwargs
    ):
        """Log API request details."""
        api_logger = self.get_logger("api")
        api_logger.info(
            "api_request",
            method=method,
            url=url,
            status_code=status_code,
            duration_seconds=duration,
            **kwargs
        )
    
    def log_agent_reasoning(
        self,
        agent: str,
        step: str,
        confidence: float,
        **kwargs
    ):
        """Log agent reasoning steps."""
        agent_logger = self.get_logger("agents")
        agent_logger.info(
            "agent_reasoning",
            agent=agent,
            step=step,
            confidence=confidence,
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        **kwargs
    ):
        """Log errors with full context."""
        error_logger = self.get_logger("errors")
        error_logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            **kwargs
        )


# Global logger instance
_logger_instance: Optional[BeatDebateLogger] = None


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    enable_console: bool = True,
    **kwargs
) -> BeatDebateLogger:
    """
    Setup the global logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Default log level
        enable_console: Whether to enable console logging
        **kwargs: Additional arguments for BeatDebateLogger
    
    Returns:
        Configured logger instance
    """
    global _logger_instance
    
    _logger_instance = BeatDebateLogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        **kwargs
    )
    
    return _logger_instance


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger for a specific component.
    
    Args:
        name: Component name
        
    Returns:
        Structured logger instance
        
    Raises:
        RuntimeError: If logging hasn't been setup
    """
    if _logger_instance is None:
        raise RuntimeError("Logging not setup. Call setup_logging() first.")
    
    return _logger_instance.get_logger(name)


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    if _logger_instance:
        _logger_instance.log_performance(operation, duration, **kwargs)


def log_api_request(method: str, url: str, status_code: int, duration: float, **kwargs):
    """Log API request details."""
    if _logger_instance:
        _logger_instance.log_api_request(method, url, status_code, duration, **kwargs)


def log_agent_reasoning(agent: str, step: str, confidence: float, **kwargs):
    """Log agent reasoning steps."""
    if _logger_instance:
        _logger_instance.log_agent_reasoning(agent, step, confidence, **kwargs)


def log_error(error: Exception, context: Dict[str, Any], **kwargs):
    """Log errors with full context."""
    if _logger_instance:
        _logger_instance.log_error(error, context, **kwargs)


def set_request_context(request_id: str, user_id: Optional[str] = None):
    """Set context for the current request."""
    if _logger_instance:
        _logger_instance.set_request_context(request_id, user_id) 