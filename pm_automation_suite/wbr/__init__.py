"""
Weekly Business Review (WBR) automation module.

Provides automated data extraction, analysis, and slide generation
for weekly business reviews.
"""

from .wbr_data_extractor import WBRDataExtractor, WBRMetric, WBRDataPackage, MetricType
from .insight_generator import InsightGenerator
from .slide_generator import SlideGenerator
from .wbr_workflow import WBRWorkflow

__all__ = [
    'WBRDataExtractor',
    'WBRMetric', 
    'WBRDataPackage',
    'MetricType',
    'InsightGenerator',
    'SlideGenerator',
    'WBRWorkflow'
]