"""
BeatDebate Main Application

This module provides the main entry point for the BeatDebate application,
integrating the FastAPI backend with the Gradio frontend for a complete
Phase 3 implementation.
"""

import asyncio
import os
import threading
import time
from typing import Optional

import gradio as gr
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup comprehensive logging FIRST - before importing modules that use it
from .utils.logging_config import setup_logging, get_logger

log_level = os.getenv("LOG_LEVEL", "INFO")
enable_console = os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"

setup_logging(
    log_dir="logs",
    log_level=log_level,
    enable_console=enable_console
)

logger = get_logger(__name__)

# Now import modules that may use logging
from .api.backend import app as fastapi_app
from .ui.chat_interface import create_chat_interface


class BeatDebateApp:
    """
    Main BeatDebate application that runs both FastAPI backend and Gradio frontend.
    
    Features:
    - FastAPI backend for 4-agent recommendation system
    - Gradio frontend for ChatGPT-style interface
    - Integrated deployment for HuggingFace Spaces
    """
    
    def __init__(
        self,
        backend_port: int = 8000,
        frontend_port: int = 7860,
        backend_host: str = "127.0.0.1",
        frontend_host: str = "0.0.0.0"
    ):
        """
        Initialize the BeatDebate application.
        
        Args:
            backend_port: Port for FastAPI backend
            frontend_port: Port for Gradio frontend
            backend_host: Host for FastAPI backend
            frontend_host: Host for Gradio frontend
        """
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.backend_host = backend_host
        self.frontend_host = frontend_host
        self.backend_url = f"http://{backend_host}:{backend_port}"
        
        # Application components
        self.fastapi_app = fastapi_app
        self.gradio_interface: Optional[gr.Blocks] = None
        self.backend_server: Optional[uvicorn.Server] = None
        
        logger.info(f"BeatDebate app initialized - Backend: {self.backend_url}")
    
    def create_gradio_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        if not self.gradio_interface:
            self.gradio_interface = create_chat_interface(self.backend_url)
        return self.gradio_interface
    
    def start_backend(self) -> None:
        """Start the FastAPI backend server in a separate thread."""
        def run_backend():
            config = uvicorn.Config(
                app=self.fastapi_app,
                host=self.backend_host,
                port=self.backend_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            self.backend_server = server
            
            logger.info(f"Starting FastAPI backend on {self.backend_url}")
            asyncio.run(server.serve())
        
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Wait for backend to start
        self._wait_for_backend()
    
    def _wait_for_backend(self, timeout: int = 30) -> None:
        """Wait for the backend to be ready."""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("Backend is ready!")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        logger.warning("Backend may not be ready - continuing anyway")
    
    def launch_frontend(
        self,
        share: bool = False,
        debug: bool = False,
        **kwargs
    ) -> None:
        """
        Launch the Gradio frontend.
        
        Args:
            share: Whether to create a public link
            debug: Whether to enable debug mode
            **kwargs: Additional arguments for Gradio launch
        """
        interface = self.create_gradio_interface()
        
        logger.info(f"Launching Gradio frontend on port {self.frontend_port}")
        
        interface.launch(
            server_name=self.frontend_host,
            server_port=self.frontend_port,
            share=share,
            debug=debug,
            **kwargs
        )
    
    def launch(
        self,
        share: bool = False,
        debug: bool = False,
        **kwargs
    ) -> None:
        """
        Launch the complete BeatDebate application.
        
        Args:
            share: Whether to create a public link
            debug: Whether to enable debug mode
            **kwargs: Additional arguments for Gradio launch
        """
        logger.info("🎵 Starting BeatDebate - AI Music Discovery System")
        
        # Start backend
        self.start_backend()
        
        # Launch frontend
        self.launch_frontend(share=share, debug=debug, **kwargs)


def create_app() -> BeatDebateApp:
    """
    Factory function to create a BeatDebate application instance.
    
    Returns:
        Configured BeatDebate application
    """
    # Get configuration from environment variables
    backend_port = int(os.getenv("BACKEND_PORT", "8000"))
    frontend_port = int(os.getenv("FRONTEND_PORT", "7860"))
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    frontend_host = os.getenv("FRONTEND_HOST", "0.0.0.0")
    
    return BeatDebateApp(
        backend_port=backend_port,
        frontend_port=frontend_port,
        backend_host=backend_host,
        frontend_host=frontend_host
    )


def main():
    """Main entry point for the application."""
    try:
        # Create and launch the application
        app = create_app()
        
        # Check if running in HuggingFace Spaces
        is_spaces = is_running_in_spaces()
        
        app.launch(
            share=not is_spaces,  # Don't create public links in Spaces
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise


def is_running_in_spaces() -> bool:
    """Check if running in HuggingFace Spaces environment."""
    return os.getenv("SPACE_ID") is not None or os.getenv("HF_SPACE_NAME") is not None


# HuggingFace Spaces compatibility
def create_gradio_app() -> gr.Blocks:
    """
    Create a Gradio app for HuggingFace Spaces deployment.
    
    This function is used when deploying to HuggingFace Spaces where
    we need to return a Gradio interface directly.
    
    Returns:
        Gradio Blocks interface
    """
    logger.info("🚀 Creating BeatDebate for HuggingFace Spaces")
    
    # Enhanced Spaces environment detection
    if is_running_in_spaces():
        space_id = os.getenv("SPACE_ID", os.getenv("HF_SPACE_NAME", "Unknown"))
        logger.info(f"🌟 Running in HuggingFace Spaces: {space_id}")
    
    # Backend configuration for Spaces
    backend_url = "http://127.0.0.1:8000"
    
    # Start backend with Spaces-optimized settings
    def start_backend_for_spaces():
        config = uvicorn.Config(
            app=fastapi_app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=False  # Reduce log noise in Spaces
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    
    backend_thread = threading.Thread(target=start_backend_for_spaces, daemon=True)
    backend_thread.start()
    
    # Wait for backend to be ready (slightly longer for Spaces)
    time.sleep(3)
    
    # Create Gradio interface with Spaces-specific configurations
    interface = create_chat_interface(backend_url)
    
    logger.info("✅ BeatDebate ready for HuggingFace Spaces deployment")
    return interface


if __name__ == "__main__":
    main() 