"""
Intelligence System for Obsidian Vault Tools

Provides proactive, context-aware intelligence across all features.
"""

from .orchestrator import IntelligenceOrchestrator
from .intent_detector import IntentDetector
from .action_executor import ActionExecutor
from .context_manager import ContextManager

__all__ = [
    'IntelligenceOrchestrator',
    'IntentDetector', 
    'ActionExecutor',
    'ContextManager'
]