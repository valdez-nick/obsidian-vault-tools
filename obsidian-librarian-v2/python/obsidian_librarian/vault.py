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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .models import Note, VaultStats, VaultConfig
try:
    from . import RustVault as librarian_core_Vault
    from . import RustVaultConfig, RustNote, RustVaultStats
    from . import RUST_BINDINGS_AVAILABLE
except ImportError:
    RUST_BINDINGS_AVAILABLE = False
    librarian_core_Vault = None

logger = structlog.get_logger(__name__)


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
        
        # Initial scan
        await self.refresh()
        
        logger.info("Vault initialized", note_count=len(self._note_cache))
    
    async def close(self) -> None:
        """Close the vault and cleanup resources."""
        logger.info("Closing vault")
        
        # Stop file watching
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        # Clear cache
        self._note_cache.clear()
    
    async def get_note(self, note_id: str) -> Optional[Note]:
        """Get a note by its ID."""
        # Check cache first
        if note_id in self._note_cache and not self._cache_dirty:
            return self._note_cache[note_id]
        
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
        
        return None
    
    async def get_all_notes(self) -> List[Note]:
        """Get all notes in the vault."""
        if not self._cache_dirty and self._note_cache:
            return list(self._note_cache.values())
        
        # Load all notes from Rust
        loop = asyncio.get_event_loop()
        rust_notes = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_all_notes()
        )
        
        notes = []
        self._note_cache.clear()
        
        for rust_note in rust_notes:
            note = Note.from_rust_note(rust_note)
            notes.append(note)
            self._note_cache[note.id] = note
        
        self._cache_dirty = False
        return notes
    
    async def get_all_note_ids(self) -> List[str]:
        """Get all note IDs in the vault."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_all_note_ids()
        )
    
    async def create_note(self, path: Path, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new note."""
        logger.debug("Creating note", path=path)
        
        loop = asyncio.get_event_loop()
        note_id = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.create_note(str(path), content, metadata or {})
        )
        
        # Update cache
        self._cache_dirty = True
        
        # Trigger callbacks
        await self._trigger_callbacks('note_created', note_id)
        
        return note_id
    
    async def update_note(self, note_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing note."""
        logger.debug("Updating note", note_id=note_id)
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.update_note(note_id, content, metadata or {})
        )
        
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
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.delete_note(note_id)
        )
        
        if success:
            # Update cache
            self._note_cache.pop(note_id, None)
            
            # Trigger callbacks
            await self._trigger_callbacks('note_deleted', note_id)
        
        return success
    
    async def move_note(self, note_id: str, new_path: Path) -> bool:
        """Move a note to a new location."""
        logger.debug("Moving note", note_id=note_id, new_path=new_path)
        
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.move_note(note_id, str(new_path))
        )
        
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
        
        loop = asyncio.get_event_loop()
        rust_notes = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.search_notes(query, limit, include_content)
        )
        
        return [Note.from_rust_note(rust_note) for rust_note in rust_notes]
    
    async def get_notes_by_tag(self, tag: str) -> List[Note]:
        """Get all notes with a specific tag."""
        loop = asyncio.get_event_loop()
        rust_notes = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_notes_by_tag(tag)
        )
        
        return [Note.from_rust_note(rust_note) for rust_note in rust_notes]
    
    async def get_linked_notes(self, note_id: str) -> List[str]:
        """Get all notes linked from the given note."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_linked_notes(note_id)
        )
    
    async def get_backlinks(self, note_id: str) -> List[str]:
        """Get all notes that link to the given note."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_backlinks(note_id)
        )
    
    async def get_orphaned_notes(self) -> List[str]:
        """Get notes with no incoming or outgoing links."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_orphaned_notes()
        )
    
    async def get_stats(self) -> VaultStats:
        """Get vault statistics."""
        loop = asyncio.get_event_loop()
        rust_stats = await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.get_stats()
        )
        
        return VaultStats.from_rust_stats(rust_stats)
    
    async def refresh(self) -> None:
        """Refresh the vault by rescanning all files."""
        logger.info("Refreshing vault")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.refresh()
        )
        
        # Clear cache
        self._note_cache.clear()
        self._cache_dirty = True
    
    async def backup(self, backup_path: Path) -> bool:
        """Create a backup of the vault."""
        logger.info("Creating vault backup", backup_path=backup_path)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.backup(str(backup_path))
        )
    
    async def validate_vault(self) -> Dict[str, Any]:
        """Validate vault integrity and return issues."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._rust_vault.validate()
        )
    
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
        self._observer = Observer()
        self._observer.schedule(
            self._file_handler, 
            str(self.path), 
            recursive=True
        )
        self._observer.start()
    
    async def _on_file_created(self, path: Path) -> None:
        """Handle file creation event."""
        logger.debug("File created", path=path)
        self._cache_dirty = True
        # The file might need time to be written completely
        await asyncio.sleep(0.1)
        await self._trigger_callbacks('note_created', str(path))
    
    async def _on_file_modified(self, path: Path) -> None:
        """Handle file modification event."""
        logger.debug("File modified", path=path)
        
        # Find note by path
        relative_path = path.relative_to(self.path)
        note_id = str(relative_path)
        
        # Clear from cache
        self._note_cache.pop(note_id, None)
        
        await self._trigger_callbacks('note_modified', note_id)
    
    async def _on_file_deleted(self, path: Path) -> None:
        """Handle file deletion event."""
        logger.debug("File deleted", path=path)
        
        relative_path = path.relative_to(self.path)
        note_id = str(relative_path)
        
        # Clear from cache
        self._note_cache.pop(note_id, None)
        
        await self._trigger_callbacks('note_deleted', note_id)
    
    async def _on_file_moved(self, old_path: Path, new_path: Path) -> None:
        """Handle file move event."""
        logger.debug("File moved", old_path=old_path, new_path=new_path)
        
        old_relative = old_path.relative_to(self.path)
        old_note_id = str(old_relative)
        
        # Clear from cache
        self._note_cache.pop(old_note_id, None)
        
        await self._trigger_callbacks('note_moved', old_note_id)
    
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