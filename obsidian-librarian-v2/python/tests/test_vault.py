"""
Unit tests for vault operations.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from obsidian_librarian.vault import Vault, VaultConfig, scan_vault_async
from obsidian_librarian.models import Note, NoteMetadata, VaultStats


@pytest.fixture
def vault_config():
    """Create test vault configuration."""
    return VaultConfig(
        cache_size=100,
        enable_file_watching=False,
        batch_size=10,
    )


@pytest.fixture
def mock_rust_vault():
    """Create a mock Rust vault instance."""
    mock = Mock()
    mock.get_all_note_ids = Mock(return_value=["note1", "note2", "note3"])
    mock.get_stats = Mock(return_value=Mock(
        note_count=3,
        total_words=1000,
        total_links=50,
        total_backlinks=30,
        total_tasks=10,
        completed_tasks=5,
        total_tags=20,
        unique_tags=15,
        orphaned_notes=1,
        total_size_bytes=50000,
    ))
    return mock


@pytest.mark.asyncio
async def test_vault_initialization(tmp_path, vault_config):
    """Test vault initialization."""
    vault = Vault(tmp_path, vault_config)
    
    assert vault.path == tmp_path
    assert vault.config == vault_config
    assert len(vault._note_cache) == 0
    assert vault._cache_dirty == True


@pytest.mark.asyncio
async def test_vault_context_manager(tmp_path, vault_config):
    """Test vault context manager."""
    with patch('obsidian_librarian.vault.librarian_core.Vault') as mock_rust:
        mock_rust.new.return_value = Mock()
        
        async with Vault(tmp_path, vault_config) as vault:
            assert vault is not None
            assert vault._rust_vault is not None


@pytest.mark.asyncio
async def test_get_all_notes(tmp_path, vault_config, mock_rust_vault):
    """Test getting all notes from vault."""
    vault = Vault(tmp_path, vault_config)
    vault._rust_vault = mock_rust_vault
    
    # Mock Rust notes
    mock_notes = []
    for i in range(3):
        mock_note = Mock()
        mock_note.id = f"note{i+1}"
        mock_note.path = f"note{i+1}.md"
        mock_note.content = f"Content {i+1}"
        mock_note.title = f"Note {i+1}"
        mock_note.tags = ["test"]
        mock_note.created = None
        mock_note.modified = None
        mock_note.word_count = 100
        mock_note.file_size = 1000
        mock_notes.append(mock_note)
    
    mock_rust_vault.get_all_notes.return_value = mock_notes
    
    notes = await vault.get_all_notes()
    
    assert len(notes) == 3
    assert all(isinstance(note, Note) for note in notes)
    assert notes[0].id == "note1"
    assert notes[1].id == "note2"
    assert notes[2].id == "note3"


@pytest.mark.asyncio
async def test_get_note_caching(tmp_path, vault_config, mock_rust_vault):
    """Test note caching behavior."""
    vault = Vault(tmp_path, vault_config)
    vault._rust_vault = mock_rust_vault
    
    mock_note = Mock()
    mock_note.id = "cached_note"
    mock_note.path = "cached.md"
    mock_note.content = "Cached content"
    mock_note.title = "Cached"
    mock_note.tags = []
    mock_note.created = None
    mock_note.modified = None
    mock_note.word_count = 10
    mock_note.file_size = 100
    
    mock_rust_vault.get_note.return_value = mock_note
    
    # First call should hit Rust
    note1 = await vault.get_note("cached_note")
    assert mock_rust_vault.get_note.call_count == 1
    
    # Mark cache as clean
    vault._cache_dirty = False
    
    # Second call should use cache
    note2 = await vault.get_note("cached_note")
    assert mock_rust_vault.get_note.call_count == 1  # No additional calls
    assert note1.id == note2.id


@pytest.mark.asyncio
async def test_create_note(tmp_path, vault_config, mock_rust_vault):
    """Test creating a new note."""
    vault = Vault(tmp_path, vault_config)
    vault._rust_vault = mock_rust_vault
    vault._trigger_callbacks = AsyncMock()
    
    mock_rust_vault.create_note.return_value = "new_note_id"
    
    note_path = tmp_path / "new_note.md"
    content = "# New Note\n\nThis is content."
    metadata = {"tags": ["new", "test"]}
    
    note_id = await vault.create_note(note_path, content, metadata)
    
    assert note_id == "new_note_id"
    assert vault._cache_dirty == True
    mock_rust_vault.create_note.assert_called_once_with(
        str(note_path), content, metadata
    )
    vault._trigger_callbacks.assert_called_once_with('note_created', 'new_note_id')


@pytest.mark.asyncio
async def test_search_notes(tmp_path, vault_config, mock_rust_vault):
    """Test searching notes."""
    vault = Vault(tmp_path, vault_config)
    vault._rust_vault = mock_rust_vault
    
    mock_results = []
    for i in range(2):
        mock_note = Mock()
        mock_note.id = f"result{i+1}"
        mock_note.path = f"result{i+1}.md"
        mock_note.content = f"Search result {i+1}"
        mock_note.title = f"Result {i+1}"
        mock_note.tags = []
        mock_note.created = None
        mock_note.modified = None
        mock_note.word_count = 50
        mock_note.file_size = 500
        mock_results.append(mock_note)
    
    mock_rust_vault.search_notes.return_value = mock_results
    
    results = await vault.search_notes("test query", limit=10)
    
    assert len(results) == 2
    assert all(isinstance(note, Note) for note in results)
    mock_rust_vault.search_notes.assert_called_once_with("test query", 10, True)


@pytest.mark.asyncio
async def test_get_stats(tmp_path, vault_config, mock_rust_vault):
    """Test getting vault statistics."""
    vault = Vault(tmp_path, vault_config)
    vault._rust_vault = mock_rust_vault
    
    stats = await vault.get_stats()
    
    assert isinstance(stats, VaultStats)
    assert stats.note_count == 3
    assert stats.total_words == 1000
    assert stats.total_links == 50


@pytest.mark.asyncio
async def test_event_callbacks(tmp_path, vault_config):
    """Test event callback system."""
    vault = Vault(tmp_path, vault_config)
    
    # Track callback calls
    callback_calls = []
    
    def sync_callback(note_id):
        callback_calls.append(('sync', note_id))
    
    async def async_callback(note_id):
        callback_calls.append(('async', note_id))
    
    # Add callbacks
    vault.add_event_callback('note_created', sync_callback)
    vault.add_event_callback('note_created', async_callback)
    
    # Trigger callbacks
    await vault._trigger_callbacks('note_created', 'test_note')
    
    assert len(callback_calls) == 2
    assert ('sync', 'test_note') in callback_calls
    assert ('async', 'test_note') in callback_calls


@pytest.mark.asyncio
async def test_scan_vault_async(tmp_path):
    """Test vault scanning utility."""
    # Create test structure
    (tmp_path / "note1.md").write_text("# Note 1\nContent")
    (tmp_path / "note2.md").write_text("# Note 2\nMore content")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "note3.md").write_text("# Note 3\nSub content")
    
    info = await scan_vault_async(tmp_path)
    
    assert info['path'] == str(tmp_path)
    assert info['exists'] == True
    assert info['note_count'] == 3
    assert info['total_size'] > 0
    assert info['last_modified'] is not None


@pytest.mark.asyncio
async def test_file_watching_debounce(tmp_path, vault_config):
    """Test file watching debounce behavior."""
    vault_config.enable_file_watching = True
    vault = Vault(tmp_path, vault_config)
    
    # Mock the file handler
    from obsidian_librarian.vault import VaultFileHandler
    handler = VaultFileHandler(vault)
    
    # Simulate rapid file changes
    test_path = str(tmp_path / "test.md")
    
    # Track scheduled events
    handler._pending_events = {}
    
    # Multiple rapid modifications should only schedule one event
    handler.on_modified(Mock(src_path=test_path, is_directory=False))
    assert len(handler._pending_events) == 1
    
    handler.on_modified(Mock(src_path=test_path, is_directory=False))
    assert len(handler._pending_events) == 1  # Still just one


def test_vault_config_to_rust():
    """Test VaultConfig conversion to Rust format."""
    config = VaultConfig(
        cache_size=500,
        max_file_size_mb=5,
        enable_file_watching=True,
    )
    
    rust_config = config.to_rust_config()
    
    assert rust_config['cache_size'] == 500
    assert rust_config['max_file_size'] == 5 * 1024 * 1024
    assert rust_config['enable_watcher'] == True
    assert 'include_patterns' in rust_config
    assert 'exclude_patterns' in rust_config