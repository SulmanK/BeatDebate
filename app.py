"""
HuggingFace Spaces entry point for BeatDebate.
This file is required by HuggingFace Spaces as the main application entry point.
"""
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import after path setup
from src.main import create_gradio_app

# Create and launch the Gradio app
app = create_gradio_app()

if __name__ == "__main__":
    app.launch() 