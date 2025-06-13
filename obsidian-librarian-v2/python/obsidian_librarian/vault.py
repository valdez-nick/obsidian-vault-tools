"""
Python vault wrapper that uses Rust bindings for high-performance operations.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, AsyncGenerator, Set, Any
from concurrent.futures import ThreadPoolExecutor

import structlog

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

from .models import Note, VaultStats, VaultConfig, NoteMetadata, WikiLink, Task
try:
    from . import RustVault as librarian_core_Vault
    from . import RustVaultConfig, RustNote, RustVaultStats
    from . import RUST_BINDINGS_AVAILABLE
except ImportError:
    RUST_BINDINGS_AVAILABLE = False
    librarian_core_Vault = None

logger = structlog.get_logger(__name__)


if WATCHDOG_AVAILABLE:
    class VaultFileHandler(FileSystemEventHandler):
        """File system event handler for vault changes."""
        
        def __init__(self, vault: 'Vault'):
            self.vault = vault
            self._debounce_delay = 0.5  # 500ms debounce
            self._pending_events: Dict[str, asyncio.Handle] = {}
        
        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith('.md'):
                self._schedule_event('modified', event.src_path)
        
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith('.md'):
                self._schedule_event('created', event.src_path)
        
        def on_deleted(self, event):
            if not event.is_directory and event.src_path.endswith('.md'):
                self._schedule_event('deleted', event.src_path)
        
        def on_moved(self, event):
            if not event.is_directory and event.src_path.endswith('.md'):
                self._schedule_event('moved', event.src_path, event.dest_path)
        
        def _schedule_event(self, event_type: str, src_path: str, dest_path: Optional[str] = None):
            """Schedule an event with debouncing."""
            # Cancel previous event for this file
            if src_path in self._pending_events:
                self._pending_events[src_path].cancel()
            
            # Schedule new event
            loop = asyncio.get_event_loop()
            handle = loop.call_later(
                self._debounce_delay,
                lambda: asyncio.create_task(self._handle_event(event_type, src_path, dest_path))
            )
            self._pending_events[src_path] = handle
        
        async def _handle_event(self, event_type: str, src_path: str, dest_path: Optional[str] = None):
            """Handle a debounced file system event."""
            try:
                # Remove from pending events
                self._pending_events.pop(src_path, None)
                
                path = Path(src_path)
                
                if event_type == 'created':
                    await self.vault._on_file_created(path)
                elif event_type == 'modified':
                    await self.vault._on_file_modified(path)
                elif event_type == 'deleted':
                    await self.vault._on_file_deleted(path)
                elif event_type == 'moved' and dest_path:
                    await self.vault._on_file_moved(path, Path(dest_path))
                    
            except Exception as e:
                logger.error("Error handling file system event", 
                            event_type=event_type, 
                            path=src_path, 
                            error=str(e))

else:
    class VaultFileHandler:
        """Dummy file system event handler when watchdog is not available."""
        
        def __init__(self, vault: 'Vault'):
            self.vault = vault
            logger.warning("File system monitoring disabled (watchdog not available)")
        
        def on_modified(self, event):
            pass
        
        def on_created(self, event):
            pass
        
        def on_deleted(self, event):
            pass
        
        def on_moved(self, event):
            pass


class Vault:
    """
    High-level Python wrapper for Obsidian vault operations.
    
    Uses Rust bindings for performance-critical operations while providing
    a convenient Python API with async support.
    """
    
    def __init__(self, vault_path: Path, config: Optional[VaultConfig] = None):
        self.path = vault_path
        self.config = config or VaultConfig()
        
        # Rust vault instance
        self._rust_vault = None
        
        # File watching
        self._observer = None
        self._file_handler = None
        
        # Threading for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Event callbacks
        self._event_callbacks: Dict[str, List[callable]] = {
            'note_created': [],
            'note_modified': [],
            'note_deleted': [],
            'note_moved': [],
        }
        
        # Cache
        self._note_cache: Dict[str, Note] = {}
        self._cache_dirty = True
        
        # Git integration - Initialize later to avoid circular import
        self._git_service = None
    
    async def initialize(self) -> None:
        """Initialize the vault and start file watching."""
        logger.info("Initializing vault", path=self.path)
        
        # Initialize Rust vault
        loop = asyncio.get_event_loop()
        if RUST_BINDINGS_AVAILABLE and librarian_core_Vault:
            self._rust_vault = await loop.run_in_executor(
                self._executor,
                lambda: librarian_core_Vault(str(self.path), RustVaultConfig())
            )
        else:
            self._rust_vault = None
            logger.warning("Running without Rust bindings - performance will be limited")
        
        # Start file watching if enabled
        if self.config.enable_file_watching:
            await self._start_file_watching()
        
        # Initialize Git service if enabled
        self.initialize_git_service()
        
        # Initialize Git repository if enabled
        if self._git_service:
            await self._git_service.initialize_repo()
        
        # Initial scan
        await self.refresh()
        
        logger.info("Vault initialized", note_count=len(self._note_cache))
    
    async def close(self) -> None:
        """Close the vault and cleanup resources."""
        logger.info("Closing vault")
        
        # Create shutdown backup if Git is enabled
        if self._git_service and self.config.enable_auto_backup:
            try:
                await self._git_service.backup(message="[Shutdown backup] Vault closed", auto=True)
            except Exception as e:
                logger.error("Failed to create shutdown backup", error=str(e))
        
        # Stop file watching
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear cache
        self._note_cache.clear()
    
    @property
    def git_service(self):
        """Get the GitService instance."""
        return self._git_service
    
    def initialize_git_service(self):
        """Initialize Git service if enabled in configuration."""
        if self.config.enable_git_integration:
            from .services.git_service import GitService, GitConfig
            
            # Create GitConfig from VaultConfig
            git_config = GitConfig(
                auto_backup_enabled=self.config.enable_auto_backup,
                auto_backup_threshold=self.config.git_auto_backup_threshold,
                auto_backup_interval=self.config.git_auto_backup_interval,
                backup_branch_prefix=self.config.git_backup_branch_prefix,
                experiment_branch_prefix=self.config.git_experiment_branch_prefix,
                commit_message_template=getattr(self.config, 'git_commit_template', 
                                              "Obsidian Librarian: {action} - {timestamp}"),
                include_stats_in_commit=True,
                default_branch=getattr(self.config, 'git_branch', 'main')
            )
            
            self._git_service = GitService(self, git_config)
            logger.info("Git service initialized", enabled=True)
        else:
            logger.info("Git service not enabled in configuration")
    
    async def get_note(self, note_id: str) -> Optional[Note]:
        """Get a note by its ID."""
        # Check cache first
        if note_id in self._note_cache and not self._cache_dirty:
            return self._note_cache[note_id]
        
        # If Rust bindings are available, use them
        if self._rust_vault:
            # Load from Rust
            loop = asyncio.get_event_loop()
            rust_note = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.get_note(note_id)
            )
            
            if rust_note:
                note = Note.from_rust_note(rust_note)
                self._note_cache[note_id] = note
                return note
        else:
            # Fallback: read note directly from file system
            note_path = self.path / note_id
            if note_path.exists() and note_path.is_file():
                note = await self._read_note_from_file(note_path)
                if note:
                    self._note_cache[note_id] = note
                    return note
        
        return None
    
    async def get_all_notes(self) -> List[Note]:
        """Get all notes in the vault."""
        if not self._cache_dirty and self._note_cache:
            return list(self._note_cache.values())
        
        notes = []
        self._note_cache.clear()
        
        if self._rust_vault:
            # Load all notes from Rust
            loop = asyncio.get_event_loop()
            rust_notes = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.get_all_notes()
            )
            
            for rust_note in rust_notes:
                note = Note.from_rust_note(rust_note)
                notes.append(note)
                self._note_cache[note.id] = note
        else:
            # Fallback: scan markdown files directly
            note_files = list(self.path.rglob("*.md"))
            for note_file in note_files:
                note = await self._read_note_from_file(note_file)
                if note:
                    notes.append(note)
                    self._note_cache[note.id] = note
        
        self._cache_dirty = False
        return notes
    
    async def get_all_note_ids(self) -> List[str]:
        """Get all note IDs in the vault."""
        if not self._rust_vault:
            # Fallback: scan markdown files directly
            note_files = list(self.path.rglob("*.md"))
            return [str(f.relative_to(self.path)) for f in note_files]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_all_note_ids()
        )
    
    async def create_note(self, path: Path, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new note."""
        logger.debug("Creating note", path=path)
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            note_id = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.create_note(str(path), content, metadata or {})
            )
        else:
            # Fallback: create file directly
            full_path = self.path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add frontmatter if metadata provided
            if metadata:
                import yaml
                frontmatter = yaml.dump(metadata, default_flow_style=False)
                content = f"---\n{frontmatter}---\n\n{content}"
            
            full_path.write_text(content, encoding='utf-8')
            note_id = str(path)
        
        # Update cache
        self._cache_dirty = True
        
        # Trigger callbacks
        await self._trigger_callbacks('note_created', note_id)
        
        return note_id
    
    async def update_note(self, note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing note."""
        logger.debug("Updating note", note_id=note_id)
        
        success = False
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.update_note(note_id, content, metadata or {})
            )
        else:
            # Fallback: update file directly
            note_path = self.path / note_id
            if note_path.exists():
                # Add frontmatter if metadata provided
                if metadata:
                    import yaml
                    frontmatter = yaml.dump(metadata, default_flow_style=False)
                    content = f"---\n{frontmatter}---\n\n{content}"
                
                note_path.write_text(content, encoding='utf-8')
                success = True
        
        if success:
            # Update cache
            if note_id in self._note_cache:
                del self._note_cache[note_id]
            
            # Trigger callbacks
            await self._trigger_callbacks('note_modified', note_id)
        
        return success
    
    async def delete_note(self, note_id: str) -> bool:
        """Delete a note."""
        logger.debug("Deleting note", note_id=note_id)
        
        success = False
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.delete_note(note_id)
            )
        else:
            # Fallback: delete file directly
            note_path = self.path / note_id
            if note_path.exists():
                note_path.unlink()
                success = True
        
        if success:
            # Update cache
            self._note_cache.pop(note_id, None)
            
            # Trigger callbacks
            await self._trigger_callbacks('note_deleted', note_id)
        
        return success
    
    async def move_note(self, note_id: str, new_path: Path) -> bool:
        """Move a note to a new location."""
        logger.debug("Moving note", note_id=note_id, new_path=new_path)
        
        success = False
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.move_note(note_id, str(new_path))
            )
        else:
            # Fallback: move file directly
            old_path = self.path / note_id
            full_new_path = self.path / new_path
            
            if old_path.exists():
                try:
                    full_new_path.parent.mkdir(parents=True, exist_ok=True)
                    old_path.rename(full_new_path)
                    success = True
                except Exception as e:
                    logger.error("Failed to move note", error=str(e))
        
        if success:
            # Update cache
            if note_id in self._note_cache:
                del self._note_cache[note_id]
            
            # Trigger callbacks
            await self._trigger_callbacks('note_moved', note_id)
        
        return success
    
    async def search_notes(
        self, 
        query: str, 
        limit: Optional[int] = None,
        include_content: bool = True,
    ) -> List[Note]:
        """Search notes using full-text search."""
        logger.debug("Searching notes", query=query, limit=limit)
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            rust_notes = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.search_notes(query, limit, include_content)
            )
            return [Note.from_rust_note(rust_note) for rust_note in rust_notes]
        else:
            # Fallback: simple search through cached notes
            all_notes = await self.get_all_notes()
            results = []
            query_lower = query.lower()
            
            for note in all_notes:
                if (query_lower in note.content.lower() or 
                    query_lower in note.title.lower() or
                    any(query_lower in tag.lower() for tag in note.tags)):
                    results.append(note)
                    if limit and len(results) >= limit:
                        break
            
            return results
    
    async def get_notes_by_tag(self, tag: str) -> List[Note]:
        """Get all notes with a specific tag."""
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            rust_notes = await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.get_notes_by_tag(tag)
            )
            return [Note.from_rust_note(rust_note) for rust_note in rust_notes]
        else:
            # Fallback: filter notes by tag
            all_notes = await self.get_all_notes()
            return [note for note in all_notes if tag in note.tags]
    
    async def get_linked_notes(self, note_id: str) -> List[str]:
        """Get all notes linked from the given note."""
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.get_linked_notes(note_id)
            )
        else:
            # Fallback: extract links from note content
            note = await self.get_note(note_id)
            if note:
                return [link.target for link in note.links]
            return []
    
    async def get_backlinks(self, note_id: str) -> List[str]:
        """Get all notes that link to the given note."""
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.get_backlinks(note_id)
            )
        else:
            # Fallback: search through all notes for backlinks
            backlinks = []
            all_notes = await self.get_all_notes()
            
            # Get the note title to search for
            target_note = await self.get_note(note_id)
            if not target_note:
                return []
            
            target_title = target_note.title
            target_path = Path(note_id).stem  # filename without extension
            
            for note in all_notes:
                if note.id != note_id:
                    for link in note.links:
                        if link.target == target_title or link.target == target_path or link.target == note_id:
                            backlinks.append(note.id)
                            break
            
            return backlinks
    
    async def get_orphaned_notes(self) -> List[str]:
        """Get notes with no incoming or outgoing links."""
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.get_orphaned_notes()
            )
        else:
            # Fallback: find notes with no links
            orphaned = []
            all_notes = await self.get_all_notes()
            
            for note in all_notes:
                # Check if note has outgoing links
                if note.links:
                    continue
                
                # Check if note has incoming links (backlinks)
                backlinks = await self.get_backlinks(note.id)
                if not backlinks:
                    orphaned.append(note.id)
            
            return orphaned
    
    async def get_stats(self) -> VaultStats:
        """Get vault statistics."""
        if not self._rust_vault:
            # Fallback: calculate basic stats
            note_ids = await self.get_all_note_ids()
            from .models import VaultStats
            return VaultStats(
                note_count=len(note_ids),
                total_words=0,  # Would need to read files to calculate
                total_links=0,  # Would need to parse files to calculate
                orphaned_notes=0,
                last_modified=datetime.utcnow(),
            )
        
        loop = asyncio.get_event_loop()
        rust_stats = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_stats()
        )
        
        return VaultStats.from_rust_stats(rust_stats)
    
    async def refresh(self) -> None:
        """Refresh the vault by rescanning all files."""
        logger.info("Refreshing vault")
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.refresh()
            )
        else:
            # Without Rust bindings, just clear cache to force reload
            logger.warning("Rust vault not available, using fallback refresh")
        
        # Clear cache
        self._note_cache.clear()
        self._cache_dirty = True
    
    async def backup(self, backup_path: Path) -> bool:
        """Create a backup of the vault."""
        logger.info("Creating vault backup", backup_path=backup_path)
        
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.backup(str(backup_path))
            )
        else:
            # Fallback: use shutil to copy vault
            try:
                import shutil
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(self.path, backup_path)
                return True
            except Exception as e:
                logger.error("Backup failed", error=str(e))
                return False
    
    async def validate_vault(self) -> Dict[str, Any]:
        """Validate vault integrity and return issues."""
        if self._rust_vault:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: self._rust_vault.validate()
            )
        else:
            # Fallback: basic validation
            issues = []
            note_count = 0
            
            for md_file in self.path.rglob("*.md"):
                note_count += 1
                try:
                    md_file.read_text(encoding='utf-8')
                except Exception as e:
                    issues.append({
                        'file': str(md_file),
                        'issue': f'Cannot read file: {str(e)}'
                    })
            
            return {
                'valid': len(issues) == 0,
                'note_count': note_count,
                'issues': issues
            }
    
    def add_event_callback(self, event_type: str, callback: callable) -> None:
        """Add a callback for vault events."""
        if event_type in self._event_callbacks:
            self._event_callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Unknown event type: {event_type}")
    
    def remove_event_callback(self, event_type: str, callback: callable) -> None:
        """Remove a callback for vault events."""
        if event_type in self._event_callbacks:
            try:
                self._event_callbacks[event_type].remove(callback)
            except ValueError:
                pass
    
    async def _start_file_watching(self) -> None:
        """Start watching the vault directory for changes."""
        logger.info("Starting file watching")
        
        self._file_handler = VaultFileHandler(self)
        
        if WATCHDOG_AVAILABLE:
            self._observer = Observer()
            self._observer.schedule(
                self._file_handler, 
                str(self.path), 
                recursive=True
            )
            self._observer.start()
        else:
            self._observer = None
    
    async def _on_file_created(self, path: Path) -> None:
        """Handle file creation event."""
        logger.debug("File created", path=path)
        self._cache_dirty = True
        # The file might need time to be written completely
        await asyncio.sleep(0.1)
        await self._trigger_callbacks('note_created', str(path))
        
        # Register change for Git auto-backup
        if self._git_service:
            self._git_service.register_change()
    
    async def _on_file_modified(self, path: Path) -> None:
        """Handle file modification event."""
        logger.debug("File modified", path=path)
        
        # Find note by path
        relative_path = path.relative_to(self.path)
        note_id = str(relative_path)
        
        # Clear from cache
        self._note_cache.pop(note_id, None)
        
        await self._trigger_callbacks('note_modified', note_id)
        
        # Register change for Git auto-backup
        if self._git_service:
            self._git_service.register_change()
    
    async def _on_file_deleted(self, path: Path) -> None:
        """Handle file deletion event."""
        logger.debug("File deleted", path=path)
        
        relative_path = path.relative_to(self.path)
        note_id = str(relative_path)
        
        # Clear from cache
        self._note_cache.pop(note_id, None)
        
        await self._trigger_callbacks('note_deleted', note_id)
        
        # Register change for Git auto-backup
        if self._git_service:
            self._git_service.register_change()
    
    async def _on_file_moved(self, old_path: Path, new_path: Path) -> None:
        """Handle file move event."""
        logger.debug("File moved", old_path=old_path, new_path=new_path)
        
        old_relative = old_path.relative_to(self.path)
        old_note_id = str(old_relative)
        
        # Clear from cache
        self._note_cache.pop(old_note_id, None)
        
        await self._trigger_callbacks('note_moved', old_note_id)
        
        # Register change for Git auto-backup
        if self._git_service:
            self._git_service.register_change()
    
    async def _trigger_callbacks(self, event_type: str, note_id: str) -> None:
        """Trigger all callbacks for an event type."""
        callbacks = self._event_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(note_id)
                else:
                    callback(note_id)
            except Exception as e:
                logger.error("Callback failed", 
                           event_type=event_type, 
                           note_id=note_id, 
                           error=str(e))
    
    async def _read_note_from_file(self, note_path: Path) -> Optional[Note]:
        """Read a note directly from the file system (fallback when Rust bindings unavailable)."""
        try:
            import re
            import yaml
            
            # Read file content
            content = note_path.read_text(encoding='utf-8')
            
            # Extract frontmatter if present
            frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
            metadata = NoteMetadata()
            
            if frontmatter_match:
                frontmatter_text = frontmatter_match.group(1)
                try:
                    frontmatter_data = yaml.safe_load(frontmatter_text) or {}
                    metadata.title = frontmatter_data.get('title')
                    metadata.tags = frontmatter_data.get('tags', [])
                    metadata.aliases = frontmatter_data.get('aliases', [])
                    metadata.custom_fields = {k: v for k, v in frontmatter_data.items() 
                                            if k not in ['title', 'tags', 'aliases']}
                except yaml.YAMLError:
                    logger.warning("Failed to parse frontmatter", path=note_path)
            
            # Extract basic information
            stat = note_path.stat()
            
            # Extract links (simple regex for wiki links)
            wiki_links = []
            for match in re.finditer(r'\[\[([^\]|]+)(?:\|([^\]]+))?\]\]', content):
                target = match.group(1)
                alias = match.group(2)
                wiki_links.append(WikiLink(target=target, alias=alias, position=match.start()))
            
            # Extract tasks
            tasks = []
            for match in re.finditer(r'^\s*[-*]\s+\[([x\s])\]\s+(.+)$', content, re.MULTILINE):
                completed = match.group(1).lower() == 'x'
                description = match.group(2)
                tasks.append(Task(description=description, completed=completed, position=match.start()))
            
            # Extract headings
            headings = []
            for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
                headings.append(match.group(2))
            
            # Calculate word count
            words = re.findall(r'\b\w+\b', content)
            word_count = len(words)
            
            # Create Note object
            note_id = str(note_path.relative_to(self.path))
            
            note = Note(
                id=note_id,
                path=note_path,
                content=content,
                metadata=metadata,
                links=wiki_links,
                tasks=tasks,
                headings=headings,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                size_bytes=stat.st_size,
                word_count=word_count,
                reading_time_minutes=word_count / 200.0  # Assuming 200 WPM
            )
            
            return note
            
        except Exception as e:
            logger.error("Failed to read note from file", path=note_path, error=str(e))
            return None
    
    # Git operations
    @property
    def git_service(self) -> Optional[Any]:
        """Get the Git service instance."""
        return self._git_service
    
    async def git_backup(self, message: Optional[str] = None) -> Optional[str]:
        """Create a Git backup of the vault."""
        if not self._git_service:
            logger.warning("Git integration not enabled")
            return None
        return await self._git_service.backup(message)
    
    async def git_restore(self, commit_hash: str) -> bool:
        """Restore the vault to a specific Git commit."""
        if not self._git_service:
            logger.warning("Git integration not enabled")
            return False
        return await self._git_service.restore(commit_hash)
    
    async def git_status(self) -> Dict[str, Any]:
        """Get Git repository status."""
        if not self._git_service:
            return {'error': 'Git integration not enabled'}
        return await self._git_service.get_status()
    
    async def git_history(self, limit: int = 10) -> List[Any]:
        """Get Git commit history."""
        if not self._git_service:
            return []
        return await self._git_service.get_history(limit)
    
    async def git_create_branch(self, name: str, checkout: bool = True) -> Optional[str]:
        """Create a new Git branch for experiments."""
        if not self._git_service:
            logger.warning("Git integration not enabled")
            return None
        return await self._git_service.create_branch(name, checkout)
    
    async def git_stash(self, message: Optional[str] = None) -> bool:
        """Stash current changes."""
        if not self._git_service:
            logger.warning("Git integration not enabled")
            return False
        return await self._git_service.stash_changes(message)
    
    async def git_rollback(self, steps: int = 1) -> bool:
        """Rollback to a previous commit by number of steps."""
        if not self._git_service:
            logger.warning("Git integration not enabled")
            return False
        return await self._git_service.rollback(steps)
    
    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class VaultManager:
    """
    Manager for multiple vaults with shared resources.
    """
    
    def __init__(self):
        self._vaults: Dict[str, Vault] = {}
        self._executor = ThreadPoolExecutor(max_workers=8)
    
    async def open_vault(self, name: str, path: Path, config: Optional[VaultConfig] = None) -> Vault:
        """Open a vault and add it to the manager."""
        if name in self._vaults:
            raise ValueError(f"Vault '{name}' is already open")
        
        vault = Vault(path, config)
        await vault.initialize()
        
        self._vaults[name] = vault
        logger.info("Opened vault", name=name, path=path)
        
        return vault
    
    async def close_vault(self, name: str) -> None:
        """Close a vault and remove it from the manager."""
        if name not in self._vaults:
            raise ValueError(f"Vault '{name}' is not open")
        
        vault = self._vaults.pop(name)
        await vault.close()
        
        logger.info("Closed vault", name=name)
    
    def get_vault(self, name: str) -> Optional[Vault]:
        """Get an open vault by name."""
        return self._vaults.get(name)
    
    def list_vaults(self) -> List[str]:
        """List names of all open vaults."""
        return list(self._vaults.keys())
    
    async def close_all(self) -> None:
        """Close all open vaults."""
        for name in list(self._vaults.keys()):
            await self.close_vault(name)
        
        self._executor.shutdown(wait=True)
    
    # Context manager support
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()


# Utility functions for vault operations
async def scan_vault_async(vault_path: Path) -> Dict[str, Any]:
    """Quickly scan a vault and return basic information."""
    info = {
        'path': str(vault_path),
        'exists': vault_path.exists(),
        'note_count': 0,
        'total_size': 0,
        'last_modified': None,
    }
    
    if not vault_path.exists():
        return info
    
    note_files = list(vault_path.rglob('*.md'))
    info['note_count'] = len(note_files)
    
    if note_files:
        total_size = sum(f.stat().st_size for f in note_files)
        info['total_size'] = total_size
        
        last_modified = max(f.stat().st_mtime for f in note_files)
        info['last_modified'] = datetime.fromtimestamp(last_modified)
    
    return info


async def find_vaults(search_path: Path) -> List[Path]:
    """Find potential Obsidian vaults in a directory tree."""
    vaults = []
    
    for path in search_path.rglob('.obsidian'):
        if path.is_dir():
            vault_path = path.parent
            if (vault_path / '.obsidian' / 'config.json').exists():
                vaults.append(vault_path)
    
    return vaults