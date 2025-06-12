# Obsidian Librarian API Reference

## Table of Contents
1. [Core Classes](#core-classes)
2. [Services](#services)
3. [Models](#models)
4. [Utilities](#utilities)
5. [Rust Bindings](#rust-bindings)

## Core Classes

### `Vault`

High-level Python wrapper for Obsidian vault operations.

```python
from obsidian_librarian import Vault
from obsidian_librarian.models import VaultConfig

vault = Vault(vault_path, config=VaultConfig())
```

#### Methods

##### `async initialize() -> None`
Initialize the vault and start file watching.

##### `async get_note(note_id: str) -> Optional[Note]`
Get a note by its ID.

**Parameters:**
- `note_id`: Unique identifier for the note

**Returns:** Note object or None if not found

##### `async get_all_notes() -> List[Note]`
Get all notes in the vault.

**Returns:** List of all notes

##### `async create_note(path: Path, content: str, metadata: Optional[Dict[str, Any]] = None) -> str`
Create a new note.

**Parameters:**
- `path`: Path for the new note
- `content`: Note content in markdown
- `metadata`: Optional frontmatter metadata

**Returns:** Note ID of created note

##### `async update_note(note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool`
Update an existing note.

**Parameters:**
- `note_id`: ID of note to update
- `content`: New content
- `metadata`: New metadata (replaces existing)

**Returns:** True if successful

##### `async search_notes(query: str, limit: Optional[int] = None, include_content: bool = True) -> List[Note]`
Search notes using full-text search.

**Parameters:**
- `query`: Search query
- `limit`: Maximum results to return
- `include_content`: Include note content in results

**Returns:** List of matching notes

##### `async get_stats() -> VaultStats`
Get vault statistics.

**Returns:** VaultStats object with metrics

### `Librarian`

Main orchestrator for all Obsidian Librarian functionality.

```python
from obsidian_librarian import Librarian
from obsidian_librarian.models import LibrarianConfig

librarian = Librarian(config)
await librarian.initialize()
```

#### Methods

##### `async execute_curator_task(task: CuratorTask) -> Dict[str, Any]`
Execute a curator task.

**Parameters:**
- `task`: CuratorTask object defining the operation

**Returns:** Result dictionary with status and details

##### `async research_and_organize(query: ResearchQuery) -> Dict[str, Any]`
Research a topic and organize findings.

**Parameters:**
- `query`: ResearchQuery object with search parameters

**Returns:** Dictionary with research results and created notes

##### `async find_duplicates(threshold: float = 0.85) -> List[DuplicateGroup]`
Find duplicate or similar notes.

**Parameters:**
- `threshold`: Similarity threshold (0.0-1.0)

**Returns:** List of duplicate groups

## Services

### `ResearchService`

Service for researching topics and organizing findings.

```python
from obsidian_librarian.services import ResearchService
from obsidian_librarian.models import ResearchConfig

service = ResearchService(config)
```

#### Methods

##### `async research(query: str, sources: Optional[List[str]] = None, max_results: Optional[int] = None) -> AsyncGenerator[ResearchResult, None]`
Research a topic across configured sources.

**Parameters:**
- `query`: Search query
- `sources`: Optional list of domains to search
- `max_results`: Maximum results to return

**Yields:** ResearchResult objects as they're found

### `AnalysisService`

Service for analyzing vault content and finding patterns.

```python
from obsidian_librarian.services import AnalysisService
from obsidian_librarian.models import AnalysisConfig

service = AnalysisService(vault, config)
```

#### Methods

##### `async find_similar_notes(threshold: float = 0.85) -> List[Tuple[str, str, float]]`
Find similar notes based on content.

**Parameters:**
- `threshold`: Similarity threshold

**Returns:** List of (note_id1, note_id2, similarity_score) tuples

##### `async analyze_note(note_id: str) -> NoteAnalysis`
Analyze a single note.

**Parameters:**
- `note_id`: ID of note to analyze

**Returns:** NoteAnalysis object with insights

### `TemplateService`

Service for managing and applying templates.

```python
from obsidian_librarian.services import TemplateService
from obsidian_librarian.models import TemplateConfig

service = TemplateService(vault, config)
```

#### Methods

##### `async apply_template(note_ids: List[str], template_name: str) -> Dict[str, bool]`
Apply a template to multiple notes.

**Parameters:**
- `note_ids`: List of note IDs
- `template_name`: Name of template to apply

**Returns:** Dictionary mapping note_id to success status

## Models

### `Note`

Represents a note in the vault.

```python
@dataclass
class Note:
    id: str
    path: Path
    content: str
    metadata: Dict[str, Any]
    links: List[WikiLink]
    tags: List[str]
    tasks: List[Task]
    created: datetime
    modified: datetime
    word_count: int
    file_size: int
```

### `Task`

Represents a task within a note.

```python
@dataclass
class Task:
    description: str
    completed: bool
    tags: List[str]
    priority: Optional[str]
    due_date: Optional[datetime]
```

### `WikiLink`

Represents a wiki-style link.

```python
@dataclass
class WikiLink:
    target: str
    alias: Optional[str]
    section: Optional[str]
```

### `VaultConfig`

Configuration for vault operations.

```python
@dataclass
class VaultConfig:
    cache_size: int = 1000
    enable_file_watching: bool = True
    exclude_patterns: List[str] = field(default_factory=list)
    index_content: bool = True
    max_file_size: int = 10_000_000  # 10MB
```

### `ResearchQuery`

Query parameters for research operations.

```python
@dataclass
class ResearchQuery:
    query: str
    sources: Optional[List[str]] = None
    max_results: Optional[int] = None
    include_summaries: bool = True
    organize_results: bool = True
```

### `CuratorTask`

Defines a curator operation.

```python
@dataclass
class CuratorTask:
    type: str  # organize, detect_duplicates, apply_template, etc.
    description: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_notes: Optional[List[str]] = None
```

## Utilities

### `scan_vault_async`

Quickly scan a vault for basic information.

```python
from obsidian_librarian.vault import scan_vault_async

info = await scan_vault_async(vault_path)
# Returns: {'path': str, 'exists': bool, 'note_count': int, ...}
```

### `find_vaults`

Find Obsidian vaults in a directory tree.

```python
from obsidian_librarian.vault import find_vaults

vaults = await find_vaults(search_path)
# Returns: List[Path]
```

## Rust Bindings

### Low-Level Access

For performance-critical operations, you can access Rust components directly:

```python
from obsidian_librarian import RustVault, RustVaultConfig

# Create Rust vault directly
config = RustVaultConfig()
rust_vault = RustVault(str(vault_path), config)

# Use Rust operations
note = rust_vault.get_note(note_id)
stats = rust_vault.get_stats()
```

### Available Rust Components

- `RustVault`: Core vault operations
- `RustVaultConfig`: Vault configuration
- `RustNote`: Note data structure
- `RustVaultStats`: Statistics structure
- `RustFileOps`: File operations
- `RustVaultEvent`: File system events
- `RustWatcherConfig`: File watcher configuration

## Error Handling

All operations may raise the following exceptions:

### `VaultError`
Base exception for vault-related errors.

```python
from obsidian_librarian.exceptions import VaultError

try:
    await vault.get_note("invalid_id")
except VaultError as e:
    print(f"Vault error: {e}")
```

### `ResearchError`
Errors during research operations.

### `TemplateError`
Errors during template operations.

### `ConfigError`
Configuration-related errors.

## Async Context Managers

Most services support async context managers:

```python
async with Vault(vault_path) as vault:
    notes = await vault.get_all_notes()
    # Vault automatically closed

async with Librarian(config) as librarian:
    await librarian.research_and_organize(query)
    # All services cleaned up
```

## Performance Considerations

### Batch Operations

For better performance with multiple operations:

```python
# Good - single batch operation
note_ids = await vault.get_all_note_ids()

# Less efficient - multiple individual calls
notes = []
for i in range(100):
    note = await vault.get_note(f"note_{i}")
    notes.append(note)

# Better - use get_all_notes with filtering
all_notes = await vault.get_all_notes()
notes = [n for n in all_notes if n.id.startswith("note_")]
```

### Concurrent Operations

Leverage asyncio for concurrent operations:

```python
import asyncio

# Concurrent research
queries = ["topic1", "topic2", "topic3"]
results = await asyncio.gather(*[
    librarian.research_and_organize(ResearchQuery(q))
    for q in queries
])

# Concurrent note operations
tasks = []
for i in range(10):
    task = vault.create_note(
        Path(f"note_{i}.md"),
        f"Content {i}",
        {"index": i}
    )
    tasks.append(task)

note_ids = await asyncio.gather(*tasks)
```

### Memory Management

For large vaults:

```python
# Stream results instead of loading all at once
async for result in research_service.research(query):
    process_result(result)  # Process one at a time

# Use pagination for large result sets
async for batch in vault.get_notes_paginated(page_size=100):
    for note in batch:
        process_note(note)
```

## Examples

### Complete Research Workflow

```python
import asyncio
from pathlib import Path
from obsidian_librarian import Librarian
from obsidian_librarian.models import LibrarianConfig, ResearchQuery

async def research_workflow():
    # Configure librarian
    config = LibrarianConfig(
        vault_path=Path("/path/to/vault"),
        research_library_path=Path("/path/to/vault/Research Library"),
    )
    
    async with Librarian(config) as librarian:
        # Research a topic
        query = ResearchQuery(
            query="quantum computing applications",
            sources=["arxiv.org", "nature.com"],
            max_results=20,
        )
        
        # Execute research and organize
        results = await librarian.research_and_organize(query)
        
        # Process results
        print(f"Found {len(results['research_results'])} results")
        print(f"Created {len(results['organized_notes'])} notes")
        
        # Find any duplicates in the new content
        duplicates = await librarian.find_duplicates(threshold=0.9)
        if duplicates:
            print(f"Found {len(duplicates)} potential duplicates")

asyncio.run(research_workflow())
```

### Custom Vault Operations

```python
from obsidian_librarian import Vault
from obsidian_librarian.models import VaultConfig

async def custom_vault_ops():
    config = VaultConfig(
        cache_size=5000,
        enable_file_watching=True,
        exclude_patterns=["*.tmp", "trash/*"],
    )
    
    async with Vault(Path("/path/to/vault"), config) as vault:
        # Add event callback
        async def on_note_created(note_id):
            print(f"New note created: {note_id}")
            note = await vault.get_note(note_id)
            if not note.metadata.get("template"):
                # Auto-apply template for new notes
                print("Applying default template...")
        
        vault.add_event_callback("note_created", on_note_created)
        
        # Keep running to handle events
        await asyncio.sleep(3600)  # Run for 1 hour

asyncio.run(custom_vault_ops())
```