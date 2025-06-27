#!/usr/bin/env python3
"""
PM Tools Module - Product Management Burnout Prevention Suite

This module provides tools for PM-specific tasks including:
- WSJF (Weighted Shortest Job First) prioritization
- Eisenhower Matrix classification  
- Burnout detection and prevention
- Enhanced daily template generation with PM insights
- Content quality analysis and standardization
"""

# Core imports
from .task_extractor import TaskExtractor
from .wsjf_analyzer import WSJFAnalyzer  
from .eisenhower_matrix import EisenhowerMatrixClassifier
from .burnout_detector import BurnoutDetector
from .content_quality_engine import ContentQualityEngine

# Enhanced features (graceful fallback if dependencies missing)
try:
    from .daily_template_generator import DailyTemplateGenerator
    DAILY_TEMPLATE_AVAILABLE = True
except ImportError:
    DAILY_TEMPLATE_AVAILABLE = False
    
    class DailyTemplateGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("Daily template generator requires additional dependencies")

# Version info
__version__ = "2.3.0"
__author__ = "Nick Valdez"

# Public API
__all__ = [
    'TaskExtractor',
    'WSJFAnalyzer', 
    'EisenhowerMatrixClassifier',
    'BurnoutDetector',
    'ContentQualityEngine',
    'DailyTemplateGenerator',
    'DAILY_TEMPLATE_AVAILABLE'
]