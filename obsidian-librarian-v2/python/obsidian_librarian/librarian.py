"""
Main Obsidian Librarian orchestrator that coordinates all services.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field

import structlog

from .vault import Vault, VaultConfig
from .services.research import ResearchService, ResearchConfig
from .services.analysis import AnalysisService, AnalysisConfig
from .services.template import TemplateService, TemplateConfig
from .sources import SourceManager
from .ai.query_processor import QueryProcessor
from .ai.content_summarizer import ContentSummarizer
from .models import LibrarianConfig, LibrarianStats
from .database import DatabaseManager, DatabaseConfig
from .database.migrations import setup_databases

logger = structlog.get_logger(__name__)


@dataclass
class LibrarianSession:
    """A librarian session with context and state."""
    session_id: str
    vault_path: Path
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # Active services
    vault: Optional[Vault] = None
    research_service: Optional[ResearchService] = None
    analysis_service: Optional[AnalysisService] = None
    template_service: Optional[TemplateService] = None
    database_manager: Optional[DatabaseManager] = None
    
    # State
    active_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ObsidianLibrarian:
    """
    Main orchestrator for the Obsidian Librarian system.
    
    Coordinates all services and provides a high-level interface for:
    - Vault management and analysis
    - Intelligent research and content discovery
    - Template application and note organization
    - Content curation and duplicate detection
    """
    
    def __init__(self, config: Optional[LibrarianConfig] = None):
        self.config = config or LibrarianConfig()
        
        # Core components
        self.query_processor = QueryProcessor()
        self.content_summarizer = ContentSummarizer()
        self.source_manager = SourceManager()
        
        # Active sessions
        self.sessions: Dict[str, LibrarianSession] = {}
        
        # Global state
        self.is_initialized = False
        self.stats = LibrarianStats()
    
    async def initialize(self) -> None:
        """Initialize the librarian system."""
        if self.is_initialized:
            return
        
        logger.info("Initializing Obsidian Librarian")
        
        # Initialize source manager
        await self.source_manager.initialize()
        
        self.is_initialized = True
        logger.info("Obsidian Librarian initialized successfully")
    
    async def create_session(
        self,
        vault_path: Path,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new librarian session for a vault.
        
        Args:
            vault_path: Path to the Obsidian vault
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if not self.is_initialized:
            await self.initialize()
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        logger.info("Creating librarian session", session_id=session_id, vault_path=vault_path)
        
        # Create session
        session = LibrarianSession(
            session_id=session_id,
            vault_path=vault_path,
        )
        
        # Initialize database layer
        db_config = DatabaseConfig(
            analytics_path=vault_path / ".obsidian-librarian" / "analytics.db",
            vector_local_path=vault_path / ".obsidian-librarian" / "vector_db",
            cache_local_path=vault_path / ".obsidian-librarian" / "cache.db",
            cache_local_fallback=True,
        )
        session.database_manager = await setup_databases(db_config)
        
        # Initialize vault
        vault_config = VaultConfig(
            enable_file_watching=self.config.enable_file_watching,
            cache_size=self.config.vault_cache_size,
        )
        session.vault = Vault(vault_path, vault_config)
        await session.vault.initialize()
        
        # Initialize services
        session.research_service = ResearchService(
            vault=session.vault,
            config=ResearchConfig(
                max_concurrent_requests=self.config.max_concurrent_requests,
                enable_content_extraction=self.config.enable_content_extraction,
            )
        )
        
        session.analysis_service = AnalysisService(
            vault=session.vault,
            config=AnalysisConfig(
                enable_quality_scoring=self.config.enable_quality_scoring,
                batch_size=self.config.analysis_batch_size,
            )
        )
        
        session.template_service = TemplateService(
            vault=session.vault,
            config=TemplateConfig(
                auto_apply=self.config.auto_apply_templates,
                template_dirs=[vault_path / "Templates"],
            )
        )
        
        # Store session
        self.sessions[session_id] = session
        
        logger.info("Librarian session created", session_id=session_id)
        return session_id
    
    async def close_session(self, session_id: str) -> None:
        """Close a librarian session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        logger.info("Closing librarian session", session_id=session_id)
        
        # Close vault
        if session.vault:
            await session.vault.close()
        
        # Close database manager
        if session.database_manager:
            await session.database_manager.close()
        
        # Remove session
        del self.sessions[session_id]
        
        logger.info("Librarian session closed", session_id=session_id)
    
    async def research(
        self,
        session_id: str,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        organize: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Perform intelligent research and organize results.
        
        Args:
            session_id: Active session ID
            query: Natural language research query
            sources: Optional list of specific sources to search
            max_results: Maximum number of results
            organize: Whether to organize results in the vault
            
        Yields:
            Research results and organization updates
        """
        session = self._get_session(session_id)
        
        logger.info("Starting research", session_id=session_id, query=query)
        
        session.active_tasks.append(f"research_{datetime.now().timestamp()}")
        
        try:
            # Collect results
            results = []
            async with session.research_service as research:
                async for result in research.research(query, sources, max_results):
                    results.append(result)
                    yield {
                        'type': 'result',
                        'data': {
                            'url': result.url,
                            'title': result.title,
                            'summary': result.summary,
                            'quality_score': result.quality_score,
                            'source': result.source,
                        }
                    }
            
            # Organize results if requested
            if organize and results:
                yield {'type': 'status', 'message': 'Organizing research results...'}
                
                organized = await session.research_service.organize_results(results, query)
                
                yield {
                    'type': 'organization',
                    'data': {
                        'organized_counts': {k: len(v) for k, v in organized.items()},
                        'total_results': len(results),
                    }
                }
            
            # Update stats
            self.stats.total_research_queries += 1
            self.stats.total_results_found += len(results)
            
            session.completed_tasks.append(session.active_tasks.pop())
            
            yield {
                'type': 'complete',
                'data': {
                    'total_results': len(results),
                    'organized': organize,
                    'query': query,
                }
            }
            
        except Exception as e:
            logger.error("Research failed", session_id=session_id, query=query, error=str(e))
            session.errors.append(str(e))
            if session.active_tasks:
                session.active_tasks.pop()
            
            yield {
                'type': 'error',
                'error': str(e),
                'query': query,
            }
    
    async def analyze_vault(
        self,
        session_id: str,
        find_duplicates: bool = True,
        quality_analysis: bool = True,
        batch_size: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Perform comprehensive vault analysis.
        
        Args:
            session_id: Active session ID
            find_duplicates: Whether to find duplicate notes
            quality_analysis: Whether to perform quality analysis
            batch_size: Optional batch size for processing
            
        Yields:
            Analysis progress and results
        """
        session = self._get_session(session_id)
        
        logger.info("Starting vault analysis", session_id=session_id)
        
        task_id = f"analysis_{datetime.now().timestamp()}"
        session.active_tasks.append(task_id)
        
        try:
            # Get all notes
            all_notes = await session.vault.get_all_note_ids()
            total_notes = len(all_notes)
            
            yield {
                'type': 'status',
                'message': f'Starting analysis of {total_notes} notes...',
                'progress': {'current': 0, 'total': total_notes}
            }
            
            # Batch analysis
            processed = 0
            analysis_results = []
            
            batch_size = batch_size or session.analysis_service.config.batch_size
            
            def progress_callback(note_id: str, current: int, total: int):
                nonlocal processed
                processed = current
            
            async for result in session.analysis_service.batch_analyze(
                all_notes, 
                progress_callback
            ):
                analysis_results.append(result)
                
                yield {
                    'type': 'progress',
                    'data': {
                        'note_id': result.note_id,
                        'quality_score': result.quality_score,
                        'similar_count': len(result.similar_notes),
                    },
                    'progress': {'current': processed, 'total': total_notes}
                }
            
            # Find duplicates if requested
            duplicate_clusters = []
            if find_duplicates:
                yield {'type': 'status', 'message': 'Finding duplicate notes...'}
                
                duplicate_clusters = await session.analysis_service.find_duplicates()
                
                yield {
                    'type': 'duplicates',
                    'data': {
                        'cluster_count': len(duplicate_clusters),
                        'clusters': [
                            {
                                'cluster_id': cluster.cluster_id,
                                'note_count': len(cluster.note_ids),
                                'confidence': cluster.confidence_score,
                                'type': cluster.cluster_type,
                            }
                            for cluster in duplicate_clusters
                        ]
                    }
                }
            
            # Calculate statistics
            vault_stats = await session.analysis_service.get_content_statistics()
            
            # Update global stats
            self.stats.total_notes_analyzed += len(analysis_results)
            self.stats.total_duplicates_found += len(duplicate_clusters)
            
            session.completed_tasks.append(session.active_tasks.pop())
            
            yield {
                'type': 'complete',
                'data': {
                    'analysis_results': len(analysis_results),
                    'duplicate_clusters': len(duplicate_clusters),
                    'vault_stats': vault_stats,
                }
            }
            
        except Exception as e:
            logger.error("Vault analysis failed", session_id=session_id, error=str(e))
            session.errors.append(str(e))
            if task_id in session.active_tasks:
                session.active_tasks.remove(task_id)
            
            yield {
                'type': 'error',
                'error': str(e),
            }
    
    async def apply_templates(
        self,
        session_id: str,
        note_ids: Optional[List[str]] = None,
        auto_detect: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply templates to notes intelligently.
        
        Args:
            session_id: Active session ID
            note_ids: Specific notes to process (default: all notes)
            auto_detect: Whether to auto-detect appropriate templates
            
        Returns:
            Template application results
        """
        session = self._get_session(session_id)
        
        logger.info("Applying templates", session_id=session_id, auto_detect=auto_detect)
        
        try:
            if auto_detect:
                # Auto-apply templates based on rules
                applications = await session.template_service.auto_apply_templates(note_ids)
            else:
                # Manual template suggestions
                applications = []
                target_notes = note_ids or await session.vault.get_all_note_ids()
                
                for note_id in target_notes[:10]:  # Limit for manual mode
                    suggestions = await session.template_service.suggest_templates_for_note(note_id)
                    if suggestions:
                        # Apply the best suggestion
                        best_template = suggestions[0]
                        app = await session.template_service.apply_template_to_note(
                            note_id, 
                            best_template.template_name
                        )
                        applications.append(app)
            
            # Update stats
            successful_applications = [app for app in applications if app.success]
            self.stats.total_templates_applied += len(successful_applications)
            
            return {
                'total_applications': len(applications),
                'successful': len(successful_applications),
                'failed': len(applications) - len(successful_applications),
                'applications': [
                    {
                        'note_id': app.note_id,
                        'template': app.template_name,
                        'success': app.success,
                        'error': app.error,
                    }
                    for app in applications
                ]
            }
            
        except Exception as e:
            logger.error("Template application failed", session_id=session_id, error=str(e))
            return {
                'error': str(e),
                'total_applications': 0,
                'successful': 0,
                'failed': 0,
            }
    
    async def curate_content(
        self,
        session_id: str,
        remove_duplicates: bool = False,
        improve_quality: bool = True,
        organize_structure: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform intelligent content curation.
        
        Args:
            session_id: Active session ID
            remove_duplicates: Whether to remove/merge duplicates
            improve_quality: Whether to suggest quality improvements
            organize_structure: Whether to organize note structure
            
        Returns:
            Curation results
        """
        session = self._get_session(session_id)
        
        logger.info("Starting content curation", session_id=session_id)
        
        curation_results = {
            'duplicates_processed': 0,
            'quality_improvements': 0,
            'structure_improvements': 0,
            'errors': [],
        }
        
        try:
            # Find and handle duplicates
            if remove_duplicates:
                duplicates = await session.analysis_service.find_duplicates()
                
                for cluster in duplicates[:5]:  # Limit to prevent accidents
                    # For now, just log what would be done
                    # In practice, you'd implement merge logic
                    logger.info("Found duplicate cluster", 
                               cluster_id=cluster.cluster_id,
                               note_count=len(cluster.note_ids))
                    curation_results['duplicates_processed'] += 1
            
            # Apply templates for structure improvement
            if organize_structure:
                template_results = await self.apply_templates(session_id, auto_detect=True)
                curation_results['structure_improvements'] = template_results['successful']
            
            # Quality analysis and recommendations
            if improve_quality:
                all_notes = await session.vault.get_all_note_ids()
                
                for note_id in all_notes[:20]:  # Sample for demonstration
                    try:
                        analysis = await session.analysis_service.analyze_note(note_id)
                        if analysis.recommendations:
                            logger.debug("Quality recommendations", 
                                        note_id=note_id,
                                        recommendations=analysis.recommendations)
                            curation_results['quality_improvements'] += 1
                    except Exception as e:
                        curation_results['errors'].append(f"Analysis failed for {note_id}: {str(e)}")
            
            return curation_results
            
        except Exception as e:
            logger.error("Content curation failed", session_id=session_id, error=str(e))
            curation_results['errors'].append(str(e))
            return curation_results
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status information for a session."""
        session = self._get_session(session_id)
        
        vault_stats = await session.vault.get_stats() if session.vault else None
        
        return {
            'session_id': session_id,
            'vault_path': str(session.vault_path),
            'started_at': session.started_at.isoformat(),
            'active_tasks': session.active_tasks,
            'completed_tasks': len(session.completed_tasks),
            'errors': len(session.errors),
            'vault_stats': {
                'note_count': vault_stats.note_count,
                'total_words': vault_stats.total_words,
                'total_links': vault_stats.total_links,
            } if vault_stats else None,
        }
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global librarian statistics."""
        source_stats = await self.source_manager.get_source_statistics()
        
        return {
            'total_sessions': len(self.sessions),
            'active_sessions': list(self.sessions.keys()),
            'global_stats': {
                'research_queries': self.stats.total_research_queries,
                'results_found': self.stats.total_results_found,
                'notes_analyzed': self.stats.total_notes_analyzed,
                'duplicates_found': self.stats.total_duplicates_found,
                'templates_applied': self.stats.total_templates_applied,
            },
            'source_stats': source_stats,
        }
    
    def _get_session(self, session_id: str) -> LibrarianSession:
        """Get a session or raise an error."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]
    
    async def close(self) -> None:
        """Close the librarian and all sessions."""
        logger.info("Closing Obsidian Librarian")
        
        # Close all sessions
        for session_id in list(self.sessions.keys()):
            await self.close_session(session_id)
        
        # Close source manager
        await self.source_manager.close()
        
        # Clear caches
        await self.content_summarizer.clear_cache()
        
        logger.info("Obsidian Librarian closed")
    
    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for single-use operations
async def analyze_vault_quick(vault_path: Path) -> Dict[str, Any]:
    """Quick vault analysis without creating a persistent session."""
    async with ObsidianLibrarian() as librarian:
        session_id = await librarian.create_session(vault_path)
        
        results = []
        async for result in librarian.analyze_vault(session_id):
            results.append(result)
        
        # Return the final complete result
        complete_results = [r for r in results if r.get('type') == 'complete']
        return complete_results[0]['data'] if complete_results else {'error': 'Analysis incomplete'}


async def research_quick(vault_path: Path, query: str) -> List[Dict[str, Any]]:
    """Quick research without creating a persistent session."""
    async with ObsidianLibrarian() as librarian:
        session_id = await librarian.create_session(vault_path)
        
        results = []
        async for result in librarian.research(session_id, query):
            if result.get('type') == 'result':
                results.append(result['data'])
        
        return results