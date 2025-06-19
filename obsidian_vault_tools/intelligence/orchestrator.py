"""
Intelligence Orchestrator

Central coordinator for the intelligence system.
"""

import asyncio
from typing import Dict, Any, Optional, Union
import logging

from .intent_detector import IntentDetector, IntentType
from .action_executor import ActionExecutor, ActionResult
from .context_manager import ContextManager

logger = logging.getLogger(__name__)


class IntelligenceOrchestrator:
    """Orchestrates intelligent interactions across the system"""
    
    def __init__(self, vault_manager=None):
        self.vault_manager = vault_manager
        self.intent_detector = IntentDetector()
        self.action_executor = ActionExecutor(vault_manager)
        self.context_manager = ContextManager(
            vault_manager.current_vault if vault_manager else None
        )
        
        # Configuration
        self.auto_execute = True  # Automatically execute detected intents
        self.require_confirmation = False  # Ask before executing
        self.learning_enabled = True  # Learn from user patterns
        
    async def process_input(self, user_input: str, 
                          context_override: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Process user input intelligently"""
        try:
            # Record the query
            self.context_manager.record_query(user_input)
            
            # Get context hints
            context = context_override or self.context_manager.get_context_hints()
            
            # Detect intent
            intent = self.intent_detector.detect_intent(user_input, context)
            
            # Log detection
            logger.info(f"Detected intent: {intent.intent_type.value} "
                       f"(confidence: {intent.confidence:.2f})")
            
            # Record detected intent
            self.context_manager.record_query(user_input, intent.intent_type.value)
            
            # Check if we should auto-execute
            if not self.auto_execute or intent.confidence < 0.6:
                return self._suggest_actions(intent)
            
            # Execute the intent
            result = await self.action_executor.execute_intent(intent)
            
            # Record the action
            self.context_manager.record_action(result.action_taken, result)
            
            # Add intelligence info to result
            result.intent = intent
            result.confidence = intent.confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return ActionResult(
                success=False,
                data=None,
                message=f"Error processing request: {str(e)}",
                action_taken="error"
            )
    
    def process_input_sync(self, user_input: str, 
                          context_override: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Synchronous wrapper for process_input"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_input(user_input, context_override))
    
    def update_context(self, menu: Optional[str] = None, **kwargs):
        """Update context information"""
        if menu:
            self.context_manager.update_menu_context(menu)
        
        # Update any additional context
        for key, value in kwargs.items():
            self.context_manager.current_context[key] = value
    
    def get_suggestions(self) -> List[str]:
        """Get contextual suggestions"""
        return self.context_manager.get_suggested_actions()
    
    def enable_auto_execution(self, enabled: bool = True):
        """Enable or disable automatic action execution"""
        self.auto_execute = enabled
        logger.info(f"Auto-execution {'enabled' if enabled else 'disabled'}")
    
    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning from patterns"""
        self.learning_enabled = enabled
        logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
    
    def _suggest_actions(self, intent) -> ActionResult:
        """Return suggested actions without executing"""
        suggestions = "\n".join([
            f"  • {action}" for action in intent.suggested_actions
        ])
        
        message = f"""🤔 I think you want to: {intent.intent_type.value}

Suggested actions:
{suggestions}

Confidence: {intent.confidence:.0%}

Type 'yes' to execute or refine your request."""
        
        return ActionResult(
            success=True,
            data={
                'intent': intent.intent_type.value,
                'suggestions': intent.suggested_actions,
                'confidence': intent.confidence
            },
            message=message,
            action_taken="suggest_only"
        )
    
    def get_help(self) -> str:
        """Get help text for the intelligence system"""
        return """🤖 Intelligent Assistant

I understand natural language and can help with:

📊 **Analysis**
• "show my most used tags"
• "analyze my vault"
• "vault health check"

🔍 **Search & Discovery**
• "find notes about python"
• "show recent files"
• "search for TODO"

📝 **Research & Creation**
• "research machine learning"
• "create daily note"
• "new project template"

🏷️ **Organization**
• "cleanup tags"
• "organize my files"
• "merge duplicate tags"

💾 **Management**
• "backup my vault"
• "export to PDF"
• "show vault statistics"

Just type what you need in natural language!"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            'total_queries': len(self.context_manager.query_history),
            'total_actions': len(self.context_manager.action_history),
            'frequent_actions': self.context_manager.get_frequent_actions(5),
            'session_duration': self._calculate_session_duration(),
            'success_rate': self._calculate_success_rate(),
        }
    
    def _calculate_session_duration(self) -> str:
        """Calculate current session duration"""
        from datetime import datetime
        start = datetime.fromisoformat(
            self.context_manager.current_context['session_start']
        )
        duration = datetime.now() - start
        
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def _calculate_success_rate(self) -> float:
        """Calculate action success rate"""
        if not self.context_manager.action_history:
            return 100.0
        
        successful = sum(
            1 for action in self.context_manager.action_history
            if action.get('success', True)
        )
        
        total = len(self.context_manager.action_history)
        return (successful / total) * 100 if total > 0 else 100.0