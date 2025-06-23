# Memory Integration Guide

## Overview

The Memory Service is a self-learning system that enhances Obsidian Vault Tools by tracking user behavior, learning preferences, and providing intelligent predictions. This guide covers how to integrate and use the memory features in your development work.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Integration Guide](#integration-guide)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)
7. [Performance Considerations](#performance-considerations)
8. [Privacy & Security](#privacy--security)

## Architecture Overview

The memory service consists of three main components:

```
obsidian_vault_tools/memory/
├── memory_service.py    # Core memory management and persistence
├── decorators.py        # Easy integration decorators
├── predictions.py       # Predictive analytics engine
└── memory_store.json    # Persistent storage (in user config dir)
```

### Data Flow

1. **Collection**: User actions are tracked via decorators
2. **Storage**: Data is persisted locally in JSON format
3. **Analysis**: Patterns are identified and scored
4. **Prediction**: Suggestions are generated based on context
5. **Feedback**: User choices refine future predictions

## Core Components

### MemoryService

The central service managing all memory operations:

```python
from obsidian_vault_tools.memory.memory_service import MemoryService

# Get singleton instance
memory = MemoryService()

# Track an event
memory.track_event('feature_used', {
    'feature': 'tag_analysis',
    'vault_path': '/path/to/vault',
    'duration': 2.5
})

# Get usage statistics
stats = memory.get_usage_stats('tag_analysis')
```

### Decorators

Simplified integration through Python decorators:

```python
from obsidian_vault_tools.memory.decorators import track_usage, learn_preferences

@track_usage
def analyze_vault(vault_path, options=None):
    """This function's usage will be automatically tracked"""
    # Your implementation
    pass

@learn_preferences
def menu_selection(choices, user_selection):
    """This will learn user's menu preferences"""
    # Your implementation
    pass
```

### Predictions Engine

Intelligent suggestion system:

```python
from obsidian_vault_tools.memory.predictions import (
    get_suggestions,
    predict_next_action,
    get_frequent_patterns
)

# Get feature suggestions
suggestions = get_suggestions('menu_navigation', context={
    'current_menu': 'main',
    'time_of_day': 'morning'
})

# Predict next likely action
next_action = predict_next_action(recent_actions=['analyze', 'backup'])

# Get frequent usage patterns
patterns = get_frequent_patterns(min_frequency=5)
```

## Integration Guide

### Step 1: Add Memory Tracking to Features

For any user-facing feature, add the `@track_usage` decorator:

```python
from obsidian_vault_tools.memory.decorators import track_usage

class TagAnalyzer:
    @track_usage
    def analyze_tags(self, vault_path, **options):
        # Existing implementation
        results = self._perform_analysis(vault_path, options)
        
        # The decorator automatically tracks:
        # - Feature name
        # - Execution time
        # - Success/failure
        # - Parameters used
        
        return results
```

### Step 2: Learn from User Choices

For interactive features, use `@learn_preferences`:

```python
from obsidian_vault_tools.memory.decorators import learn_preferences

class MenuNavigator:
    @learn_preferences
    def display_menu(self, menu_items):
        # Show menu with predictions
        suggestions = get_suggestions('menu_' + self.current_menu)
        
        if suggestions:
            print("\nSuggested actions:")
            for i, suggestion in enumerate(suggestions[:3]):
                print(f"  {i+1}. {suggestion}")
        
        # Display regular menu
        for idx, item in enumerate(menu_items):
            print(f"{idx+1}. {item}")
        
        choice = input("Select: ")
        return choice
```

### Step 3: Provide Predictive Features

Enhance user experience with predictions:

```python
from obsidian_vault_tools.memory.predictions import predict_next_action

class VaultManager:
    def suggest_next_action(self):
        recent = self.get_recent_actions()
        suggestion = predict_next_action(recent)
        
        if suggestion:
            response = input(f"Would you like to {suggestion}? (y/n): ")
            if response.lower() == 'y':
                self.execute_action(suggestion)
```

## API Reference

### MemoryService Methods

#### `track_event(event_type: str, data: dict) -> None`
Track a custom event with associated data.

```python
memory.track_event('custom_action', {
    'action': 'bulk_rename',
    'files_affected': 42,
    'time_taken': 3.2
})
```

#### `get_usage_stats(feature: str) -> dict`
Get usage statistics for a specific feature.

```python
stats = memory.get_usage_stats('tag_analysis')
# Returns: {
#     'total_uses': 45,
#     'average_duration': 2.3,
#     'last_used': '2024-01-15T10:30:00',
#     'success_rate': 0.98
# }
```

#### `clear_memory() -> None`
Clear all stored memory data.

```python
memory.clear_memory()  # Use with caution!
```

### Decorator Parameters

#### `@track_usage(category: str = None, include_params: bool = True)`
- `category`: Override automatic category detection
- `include_params`: Whether to track function parameters

```python
@track_usage(category='analysis', include_params=False)
def sensitive_analysis(data):
    # Parameters won't be tracked
    pass
```

#### `@learn_preferences(context_keys: list = None)`
- `context_keys`: Additional context to track

```python
@learn_preferences(context_keys=['time_of_day', 'vault_size'])
def smart_suggestion(options, context):
    pass
```

### Prediction Functions

#### `get_suggestions(context: str, limit: int = 5) -> list`
Get suggestions for a given context.

#### `predict_next_action(recent_actions: list) -> str`
Predict the next likely action based on recent history.

#### `get_frequent_patterns(min_frequency: int = 3) -> list`
Get frequently occurring action patterns.

## Usage Examples

### Example 1: Smart Command Completion

```python
from obsidian_vault_tools.memory.predictions import get_suggestions

class CommandInterface:
    def autocomplete(self, partial_command):
        # Get suggestions based on command history
        suggestions = get_suggestions('commands', context={
            'partial': partial_command,
            'current_dir': os.getcwd()
        })
        
        return [s for s in suggestions if s.startswith(partial_command)]
```

### Example 2: Adaptive Menu Ordering

```python
from obsidian_vault_tools.memory.decorators import track_usage
from obsidian_vault_tools.memory.predictions import get_frequent_patterns

class AdaptiveMenu:
    @track_usage
    def display_menu(self, items):
        # Get frequently used items
        frequent = get_frequent_patterns()
        
        # Reorder menu with frequent items first
        ordered_items = []
        for pattern in frequent:
            if pattern['action'] in items:
                ordered_items.append(pattern['action'])
                items.remove(pattern['action'])
        
        # Add remaining items
        ordered_items.extend(items)
        
        return ordered_items
```

### Example 3: Workflow Automation

```python
from obsidian_vault_tools.memory.predictions import predict_next_action

class WorkflowAutomation:
    def suggest_workflow(self):
        recent = self.get_recent_5_actions()
        
        # Build workflow chain
        workflow = []
        for _ in range(3):  # Predict next 3 actions
            next_action = predict_next_action(recent)
            if next_action and next_action not in workflow:
                workflow.append(next_action)
                recent.append(next_action)
        
        if workflow:
            print(f"Suggested workflow: {' → '.join(workflow)}")
            if input("Execute? (y/n): ").lower() == 'y':
                self.execute_workflow(workflow)
```

## Best Practices

### 1. Respect User Privacy

- Never track sensitive content (file contents, personal notes)
- Only track metadata and usage patterns
- Provide clear opt-out mechanisms

```python
from obsidian_vault_tools.config import get_config

def should_track():
    config = get_config()
    return config.get('enable_memory', True)

@track_usage
def my_feature():
    if not should_track():
        return
    # Feature implementation
```

### 2. Meaningful Context

Include relevant context for better predictions:

```python
memory.track_event('feature_used', {
    'feature': 'organize_files',
    'vault_size': 'large',  # > 1000 files
    'time_of_day': 'evening',
    'day_of_week': 'sunday',
    'organization_type': 'by_date'
})
```

### 3. Graceful Degradation

Always handle memory service failures gracefully:

```python
try:
    suggestions = get_suggestions('menu_navigation')
except Exception as e:
    logging.warning(f"Memory service unavailable: {e}")
    suggestions = []  # Fall back to default behavior
```

### 4. Efficient Tracking

Avoid tracking high-frequency events:

```python
# Bad: Tracks every keystroke
@track_usage
def on_keypress(key):
    pass

# Good: Track meaningful actions
@track_usage
def execute_command(command):
    pass
```

## Performance Considerations

### Memory Usage

The memory service uses an in-memory cache with periodic persistence:

- Cache size: ~1MB for typical usage
- Persistence interval: Every 5 minutes or on exit
- History limit: Last 1000 events per category

### Optimization Tips

1. **Batch Operations**: Track batch operations as single events
2. **Async Tracking**: Use async tracking for non-blocking operation
3. **Selective Tracking**: Only track user-initiated actions

```python
from obsidian_vault_tools.memory.decorators import track_usage_async

@track_usage_async  # Non-blocking tracking
def long_running_operation():
    # This won't block on memory operations
    pass
```

### Configuration

Tune memory service via configuration:

```yaml
# ~/.obsidian-tools/config.yaml
memory:
  enabled: true
  max_events: 1000
  persistence_interval: 300  # seconds
  prediction_confidence: 0.7
  cache_size_mb: 2
```

## Privacy & Security

### Data Storage

- All data stored locally in `~/.obsidian-tools/memory/`
- No network transmission of memory data
- Encrypted storage option available

### Data Collected

The memory service tracks:
- Feature usage frequency
- Execution duration
- Success/failure status
- Time of usage
- Basic context (no content)

### User Control

Users can:
- Disable memory service entirely
- Clear memory at any time
- Export their data
- Configure retention periods

```python
# Disable memory service
ovt config set memory.enabled false

# Clear all memory
ovt memory clear

# Export memory data
ovt memory export memory_backup.json
```

### Compliance

The memory service is designed with privacy in mind:
- GDPR compliant (local storage, user control)
- No PII collection
- Transparent data usage
- Easy data deletion

## Troubleshooting

### Common Issues

1. **Memory not persisting**
   - Check write permissions on config directory
   - Verify disk space availability

2. **Poor predictions**
   - Allow time for learning (minimum 50 events)
   - Check if context is consistent

3. **Performance impact**
   - Reduce max_events in configuration
   - Disable async tracking if needed

### Debug Mode

Enable debug logging for memory service:

```python
import logging
logging.getLogger('obsidian_vault_tools.memory').setLevel(logging.DEBUG)
```

### Memory Statistics

View memory service statistics:

```bash
ovt memory stats

# Output:
# Memory Service Statistics:
# - Total events: 1,234
# - Features tracked: 15
# - Prediction accuracy: 78%
# - Storage used: 1.2 MB
```

## Future Enhancements

Planned improvements to the memory service:

1. **Machine Learning Integration**: Advanced pattern recognition
2. **Collaborative Filtering**: Learn from community patterns
3. **Export/Import**: Share workflows with others
4. **Visual Analytics**: Dashboard for usage patterns
5. **Plugin API**: Allow third-party integrations

---

For more information or to contribute to the memory service development, see the [main documentation](../README.md) or [open an issue](https://github.com/yourusername/obsidian-vault-tools/issues).