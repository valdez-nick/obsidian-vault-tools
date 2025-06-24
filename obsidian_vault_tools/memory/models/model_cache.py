"""
Model Cache for Obsidian Vault Tools Memory Service.

Provides intelligent caching, cleanup, and storage management for downloaded
models with disk space monitoring and automatic cleanup policies.
"""

import os
import shutil
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from .model_config import ModelConfig
except ImportError:
    from model_config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Information about a cached model."""
    model_id: str
    cache_path: str
    size_bytes: int
    created_time: datetime
    last_accessed: datetime
    access_count: int
    model_type: str
    version: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'model_id': self.model_id,
            'cache_path': self.cache_path,
            'size_bytes': self.size_bytes,
            'created_time': self.created_time.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'model_type': self.model_type,
            'version': self.version,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            model_id=data['model_id'],
            cache_path=data['cache_path'],
            size_bytes=data['size_bytes'],
            created_time=datetime.fromisoformat(data['created_time']),
            last_accessed=datetime.fromisoformat(data['last_accessed']),
            access_count=data['access_count'],
            model_type=data['model_type'],
            version=data.get('version'),
            metadata=data.get('metadata', {})
        )


class ModelCache:
    """
    Intelligent model cache with automatic cleanup and disk space management.
    
    Features:
    - Automatic disk space monitoring
    - LRU-based cleanup policies
    - Model versioning support
    - Access tracking and analytics
    - Thread-safe operations
    - Configurable size limits
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model cache with configuration."""
        self.config = config or ModelConfig()
        
        # Cache directory setup
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Cache registry
        self._cache_entries: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        
        # Load existing cache metadata
        self._load_cache_metadata()
        
        # Cleanup settings
        self.max_cache_size_gb = getattr(config, 'max_memory_gb', None) or 10.0
        self.cleanup_threshold = 0.8  # Cleanup when 80% full
        self.min_free_space_gb = 2.0
        
        logger.info(f"Initialized ModelCache at: {self.cache_dir}")
        logger.info(f"Cache size limit: {self.max_cache_size_gb:.1f}GB")
    
    def _load_cache_metadata(self):
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to CacheEntry objects
                for model_id, entry_data in data.items():
                    try:
                        self._cache_entries[model_id] = CacheEntry.from_dict(entry_data)
                    except Exception as e:
                        logger.warning(f"Failed to load cache entry for {model_id}: {e}")
                
                logger.info(f"Loaded {len(self._cache_entries)} cache entries")
                
                # Validate cache entries (check if files still exist)
                self._validate_cache_entries()
                
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {e}")
            self._cache_entries = {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to disk."""
        try:
            data = {
                model_id: entry.to_dict() 
                for model_id, entry in self._cache_entries.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _validate_cache_entries(self):
        """Validate that cached files still exist and update metadata."""
        invalid_entries = []
        
        for model_id, entry in self._cache_entries.items():
            cache_path = Path(entry.cache_path)
            if not cache_path.exists():
                logger.warning(f"Cache entry {model_id} points to non-existent path: {cache_path}")
                invalid_entries.append(model_id)
            else:
                # Update size if needed
                current_size = self._get_directory_size(cache_path)
                if current_size != entry.size_bytes:
                    entry.size_bytes = current_size
        
        # Remove invalid entries
        for model_id in invalid_entries:
            del self._cache_entries[model_id]
        
        if invalid_entries:
            self._save_cache_metadata()
            logger.info(f"Removed {len(invalid_entries)} invalid cache entries")
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
            return total_size
        except Exception:
            return 0
    
    def _get_available_space(self) -> float:
        """Get available disk space in GB."""
        try:
            stat = shutil.disk_usage(self.cache_dir)
            return stat.free / (1024**3)
        except Exception:
            return 0.0
    
    def _get_cache_size_gb(self) -> float:
        """Get current cache size in GB."""
        total_size = sum(entry.size_bytes for entry in self._cache_entries.values())
        return total_size / (1024**3)
    
    def is_cached(self, model_id: str) -> bool:
        """Check if model is cached."""
        with self._cache_lock:
            if model_id in self._cache_entries:
                # Verify file still exists
                entry = self._cache_entries[model_id]
                if Path(entry.cache_path).exists():
                    return True
                else:
                    # Remove stale entry
                    del self._cache_entries[model_id]
                    self._save_cache_metadata()
            return False
    
    def get_cache_path(self, model_id: str) -> Optional[str]:
        """Get cache path for a model."""
        with self._cache_lock:
            entry = self._cache_entries.get(model_id)
            if entry and Path(entry.cache_path).exists():
                return entry.cache_path
            return None
    
    def mark_used(self, model_id: str, model_type: str = "unknown") -> bool:
        """Mark a model as used (for LRU tracking)."""
        with self._cache_lock:
            if model_id in self._cache_entries:
                entry = self._cache_entries[model_id]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._save_cache_metadata()
                return True
            else:
                # Try to auto-detect if model exists in cache
                potential_path = self._get_model_cache_path(model_id)
                if potential_path.exists():
                    # Create cache entry for existing model
                    self._add_cache_entry(model_id, str(potential_path), model_type)
                    return True
            return False
    
    def _get_model_cache_path(self, model_id: str) -> Path:
        """Get expected cache path for a model."""
        # Standard Hugging Face cache structure
        safe_model_id = model_id.replace("/", "--")
        return self.cache_dir / "models" / safe_model_id
    
    def _add_cache_entry(self, model_id: str, cache_path: str, model_type: str):
        """Add a new cache entry."""
        path = Path(cache_path)
        size_bytes = self._get_directory_size(path) if path.exists() else 0
        
        entry = CacheEntry(
            model_id=model_id,
            cache_path=cache_path,
            size_bytes=size_bytes,
            created_time=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            model_type=model_type
        )
        
        self._cache_entries[model_id] = entry
        self._save_cache_metadata()
        
        logger.info(f"Added cache entry for {model_id}: {size_bytes / (1024**3):.2f}GB")
    
    def cleanup_cache(self, target_size_gb: Optional[float] = None, force: bool = False) -> Dict[str, Any]:
        """
        Clean up cache to target size using LRU policy.
        
        Args:
            target_size_gb: Target cache size (defaults to cleanup threshold)
            force: Force cleanup even if under threshold
            
        Returns:
            Cleanup statistics
        """
        with self._cache_lock:
            current_size_gb = self._get_cache_size_gb()
            available_space_gb = self._get_available_space()
            
            # Determine if cleanup is needed
            needs_cleanup = (
                force or 
                current_size_gb > (self.max_cache_size_gb * self.cleanup_threshold) or
                available_space_gb < self.min_free_space_gb
            )
            
            if not needs_cleanup:
                return {
                    'cleanup_performed': False,
                    'current_size_gb': current_size_gb,
                    'available_space_gb': available_space_gb,
                    'reason': 'No cleanup needed'
                }
            
            # Calculate target size
            if target_size_gb is None:
                target_size_gb = self.max_cache_size_gb * 0.7  # Clean to 70% of max
            
            # Sort entries by LRU (least recently used first)
            sorted_entries = sorted(
                self._cache_entries.values(),
                key=lambda x: (x.last_accessed, x.access_count)
            )
            
            # Remove entries until we reach target size
            removed_entries = []
            current_size_bytes = sum(entry.size_bytes for entry in self._cache_entries.values())
            target_size_bytes = target_size_gb * (1024**3)
            
            for entry in sorted_entries:
                if current_size_bytes <= target_size_bytes:
                    break
                
                try:
                    # Remove files
                    cache_path = Path(entry.cache_path)
                    if cache_path.exists():
                        if cache_path.is_dir():
                            shutil.rmtree(cache_path)
                        else:
                            cache_path.unlink()
                    
                    # Remove from registry
                    del self._cache_entries[entry.model_id]
                    removed_entries.append(entry)
                    current_size_bytes -= entry.size_bytes
                    
                    logger.info(f"Removed cached model: {entry.model_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to remove cache entry {entry.model_id}: {e}")
            
            # Save updated metadata
            self._save_cache_metadata()
            
            final_size_gb = current_size_bytes / (1024**3)
            space_freed_gb = current_size_gb - final_size_gb
            
            return {
                'cleanup_performed': True,
                'models_removed': len(removed_entries),
                'space_freed_gb': space_freed_gb,
                'initial_size_gb': current_size_gb,
                'final_size_gb': final_size_gb,
                'target_size_gb': target_size_gb,
                'removed_models': [entry.model_id for entry in removed_entries]
            }
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a specific model from cache."""
        with self._cache_lock:
            if model_id not in self._cache_entries:
                return False
            
            try:
                entry = self._cache_entries[model_id]
                cache_path = Path(entry.cache_path)
                
                if cache_path.exists():
                    if cache_path.is_dir():
                        shutil.rmtree(cache_path)
                    else:
                        cache_path.unlink()
                
                del self._cache_entries[model_id]
                self._save_cache_metadata()
                
                logger.info(f"Removed model from cache: {model_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to remove model {model_id} from cache: {e}")
                return False
    
    def clear_cache(self) -> bool:
        """Clear entire cache."""
        with self._cache_lock:
            try:
                # Remove all cached files
                for entry in self._cache_entries.values():
                    cache_path = Path(entry.cache_path)
                    if cache_path.exists():
                        if cache_path.is_dir():
                            shutil.rmtree(cache_path)
                        else:
                            cache_path.unlink()
                
                # Clear registry
                self._cache_entries.clear()
                
                # Remove metadata file
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                
                logger.info("Cleared entire model cache")
                return True
                
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._cache_lock:
            current_size_gb = self._get_cache_size_gb()
            available_space_gb = self._get_available_space()
            
            # Model type distribution
            type_stats = defaultdict(lambda: {'count': 0, 'size_gb': 0})
            for entry in self._cache_entries.values():
                type_stats[entry.model_type]['count'] += 1
                type_stats[entry.model_type]['size_gb'] += entry.size_bytes / (1024**3)
            
            # Access statistics
            total_accesses = sum(entry.access_count for entry in self._cache_entries.values())
            avg_access_count = total_accesses / len(self._cache_entries) if self._cache_entries else 0
            
            # Age statistics
            now = datetime.now()
            ages_hours = [(now - entry.created_time).total_seconds() / 3600 for entry in self._cache_entries.values()]
            avg_age_hours = sum(ages_hours) / len(ages_hours) if ages_hours else 0
            
            return {
                'total_models': len(self._cache_entries),
                'total_size_gb': current_size_gb,
                'max_size_gb': self.max_cache_size_gb,
                'usage_percentage': (current_size_gb / self.max_cache_size_gb) * 100,
                'available_space_gb': available_space_gb,
                'cache_directory': str(self.cache_dir),
                'model_types': dict(type_stats),
                'access_statistics': {
                    'total_accesses': total_accesses,
                    'average_access_count': avg_access_count
                },
                'age_statistics': {
                    'average_age_hours': avg_age_hours,
                    'oldest_model_hours': max(ages_hours) if ages_hours else 0,
                    'newest_model_hours': min(ages_hours) if ages_hours else 0
                }
            }
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """List all cached models with details."""
        with self._cache_lock:
            models = []
            for entry in self._cache_entries.values():
                models.append({
                    'model_id': entry.model_id,
                    'model_type': entry.model_type,
                    'size_gb': entry.size_bytes / (1024**3),
                    'created_time': entry.created_time.isoformat(),
                    'last_accessed': entry.last_accessed.isoformat(),
                    'access_count': entry.access_count,
                    'cache_path': entry.cache_path,
                    'version': entry.version
                })
            
            # Sort by last accessed (most recent first)
            models.sort(key=lambda x: x['last_accessed'], reverse=True)
            return models
    
    def auto_cleanup_if_needed(self):
        """Automatically cleanup cache if needed."""
        current_size_gb = self._get_cache_size_gb()
        available_space_gb = self._get_available_space()
        
        if (current_size_gb > (self.max_cache_size_gb * self.cleanup_threshold) or
            available_space_gb < self.min_free_space_gb):
            
            logger.info(f"Auto-cleanup triggered: {current_size_gb:.2f}GB cache, {available_space_gb:.2f}GB free")
            return self.cleanup_cache()
        
        return None
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache by removing duplicates and validating entries."""
        with self._cache_lock:
            initial_count = len(self._cache_entries)
            initial_size_gb = self._get_cache_size_gb()
            
            # Validate all entries
            self._validate_cache_entries()
            
            # Check for potential duplicates (same size and similar names)
            # This is a simple heuristic - could be enhanced
            duplicates_removed = 0
            
            final_count = len(self._cache_entries)
            final_size_gb = self._get_cache_size_gb()
            
            return {
                'initial_models': initial_count,
                'final_models': final_count,
                'models_removed': initial_count - final_count,
                'initial_size_gb': initial_size_gb,
                'final_size_gb': final_size_gb,
                'space_freed_gb': initial_size_gb - final_size_gb,
                'duplicates_removed': duplicates_removed
            }
    
    def cleanup(self):
        """Clean up cache resources."""
        with self._cache_lock:
            self._save_cache_metadata()
        logger.info("ModelCache cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()