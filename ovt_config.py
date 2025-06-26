#!/usr/bin/env python3
"""
Configuration for Obsidian Vault Tools startup behavior
"""

import os
import warnings
import logging

# Suppress optional dependency warnings on startup
SUPPRESS_OPTIONAL_WARNINGS = os.environ.get('OVT_SUPPRESS_WARNINGS', 'true').lower() == 'true'

def configure_warnings():
    """Configure warning filters for cleaner startup"""
    if SUPPRESS_OPTIONAL_WARNINGS:
        # Suppress specific warnings
        warnings.filterwarnings("ignore", message="transformers/torch library not available")
        warnings.filterwarnings("ignore", message="FAISS not available")
        warnings.filterwarnings("ignore", message="sentence-transformers not available")
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        
        # Configure logging to suppress warnings from model modules at startup
        logging.getLogger('models.transformer_adapter').setLevel(logging.ERROR)
        logging.getLogger('models.embedding_adapter').setLevel(logging.ERROR)

# Apply configuration when module is imported
configure_warnings()