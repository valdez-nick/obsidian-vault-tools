"""
Performance benchmarks for directory organization operations.

Benchmarks various organization operations including:
- Content classification
- Directory routing
- Pattern learning
- Rule evaluation  
- Batch organization
"""

import asyncio
import time
import statistics
import random
import string
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from obsidian_librarian import Vault
from obsidian_librarian.models import LibrarianConfig, VaultConfig, Note, NoteMetadata
from obsidian_librarian.services.auto_organizer import (
    AutoOrganizer,
    ContentClassifier, 
    DirectoryRouter,
    OrganizationLearner,
    RuleEngine,
    OrganizationRule,
    OrganizationAction,
    ClassificationConfidence,
)
from obsidian_librarian.ai import ContentAnalyzer, EmbeddingService


class OrganizationBenchmark:
    """Benchmark directory organization operations."""
    
    def __init__(self, num_notes: int = 1000):
        self.num_notes = num_notes
        self.results: Dict[str, List[float]] = {}
        
        # Note categories for testing
        self.categories = {
            "daily": {
                "pattern": r"\d{4}-\d{2}-\d{2}",
                "template": "# {date}\n\nDaily note for {date}.\n\n## Tasks\n- [ ] Task 1\n\n#daily",
                "target": "Daily Notes/{year}/{month}",
            },
            "meeting": {
                "keywords": ["meeting", "agenda", "attendees", "minutes"],
                "template": "# {title} Meeting\n\nDate: {date}\nAttendees: {attendees}\n\n## Agenda\n{agenda}\n\n#meeting",
                "target": "Meetings/{year}/{month}",
            },
            "project": {
                "keywords": ["project", "milestone", "deliverable", "sprint"],
                "template": "# Project {name}\n\n## Status\n{status}\n\n## Milestones\n{milestones}\n\n#project",
                "target": "Projects/{name}",
            },
            "research": {
                "keywords": ["research", "study", "analysis", "findings"],
                "template": "# Research: {topic}\n\n## Abstract\n{abstract}\n\n## Findings\n{findings}\n\n#research",
                "target": "Research/{topic}",
            },
            "general": {
                "keywords": [],
                "template": "# {title}\n\n{content}\n\n#{tag}",
                "target": "Notes",
            }
        }
    
    async def setup(self):
        """Set up test vault with various note types."""
        self.tmpdir = tempfile.mkdtemp()
        self.vault_path = Path(self.tmpdir) / "bench_vault"
        self.vault_path.mkdir()
        
        # Create vault structure
        (self.vault_path / ".obsidian").mkdir()
        
        # Create directories
        directories = [
            "Daily Notes/2024/01",
            "Meetings/2024/01", 
            "Projects/Alpha",
            "Projects/Beta",
            "Research/ML",
            "Research/Security",
            "Notes",
            "Archive",
            "Templates",
            "Unsorted",
        ]
        
        for dir_path in directories:
            (self.vault_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Generate notes
        print(f"Creating {self.num_notes} notes for organization benchmark...")
        self.notes_metadata = {}
        
        for i in range(self.num_notes):
            note_type = self._choose_note_type(i)
            content, metadata = self._generate_note(i, note_type)
            
            # Place some notes correctly, others incorrectly
            if i % 3 == 0:
                # Misplaced note
                path = self.vault_path / "Unsorted" / f"note_{i:04d}.md"
            else:
                # Correctly placed (for testing detection)
                if note_type == "daily":
                    date = datetime(2024, 1, (i % 30) + 1)
                    path = self.vault_path / "Daily Notes/2024/01" / f"{date.strftime('%Y-%m-%d')}.md"
                elif note_type == "meeting":
                    path = self.vault_path / "Meetings/2024/01" / f"meeting_{i:04d}.md"
                elif note_type == "project":
                    project = "Alpha" if i % 2 == 0 else "Beta"
                    path = self.vault_path / f"Projects/{project}" / f"note_{i:04d}.md"
                elif note_type == "research":
                    topic = "ML" if i % 2 == 0 else "Security"
                    path = self.vault_path / f"Research/{topic}" / f"note_{i:04d}.md"
                else:
                    path = self.vault_path / "Notes" / f"note_{i:04d}.md"
            
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            self.notes_metadata[path.name] = metadata
        
        # Initialize services
        self.config = LibrarianConfig(
            vault_path=self.vault_path,
            enable_ai_features=False,  # Disable for performance testing
            auto_organization={
                "enabled": True,
                "confidence_threshold": 0.7,
                "watch_mode": False,
            }
        )
        
        vault_config = VaultConfig(
            cache_size=10000,
            enable_file_watching=False,
        )
        
        self.vault = Vault(self.vault_path, vault_config)
        await self.vault.initialize()
        
        # Mock AI services for consistent benchmarking
        self.mock_content_analyzer = Mock(spec=ContentAnalyzer)
        self.mock_embedding_service = Mock(spec=EmbeddingService)
        self.mock_query_processor = Mock()
        
        # Setup mock responses
        async def mock_analyze(content):
            result = Mock()
            # Extract keywords from content
            words = content.lower().split()
            result.keywords = [w for w in words if len(w) > 5][:10]
            result.topics = self._detect_topics(content)
            result.entities = []
            return result
        
        self.mock_content_analyzer.analyze_content = AsyncMock(side_effect=mock_analyze)
        
        # Initialize organizer
        self.organizer = AutoOrganizer(
            self.vault,
            self.mock_content_analyzer,
            self.mock_embedding_service,
            self.mock_query_processor,
            self.config
        )
    
    def _choose_note_type(self, index: int) -> str:
        """Choose note type based on distribution."""
        # Realistic distribution
        if index % 7 == 0:
            return "daily"
        elif index % 5 == 0:
            return "meeting"
        elif index % 4 == 0:
            return "project"
        elif index % 6 == 0:
            return "research"
        else:
            return "general"
    
    def _generate_note(self, index: int, note_type: str) -> tuple:
        """Generate note content and metadata."""
        category = self.categories[note_type]
        
        if note_type == "daily":
            date = datetime(2024, 1, (index % 30) + 1)
            content = category["template"].format(
                date=date.strftime("%Y-%m-%d")
            )
            metadata = {
                "type": "daily",
                "date": date,
            }
            
        elif note_type == "meeting":
            content = category["template"].format(
                title=f"Team Sync {index}",
                date="2024-01-15",
                attendees="Alice, Bob, Carol",
                agenda="1. Status updates\n2. Planning"
            )
            metadata = {
                "type": "meeting",
                "attendees": ["Alice", "Bob", "Carol"],
            }
            
        elif note_type == "project":
            project_name = f"Project-{chr(65 + (index % 26))}"  # A-Z
            content = category["template"].format(
                name=project_name,
                status="In Progress",
                milestones="- [ ] Phase 1\n- [ ] Phase 2"
            )
            metadata = {
                "type": "project",
                "project": project_name,
            }
            
        elif note_type == "research":
            topics = ["Machine Learning", "Security", "Performance", "Architecture"]
            topic = topics[index % len(topics)]
            content = category["template"].format(
                topic=topic,
                abstract="Research into advanced techniques...",
                findings="1. Finding one\n2. Finding two"
            )
            metadata = {
                "type": "research",
                "topic": topic,
            }
            
        else:
            content = category["template"].format(
                title=f"General Note {index}",
                content="Some general content here.",
                tag="notes"
            )
            metadata = {
                "type": "general",
            }
        
        return content, metadata
    
    def _detect_topics(self, content: str) -> List[str]:
        """Simple topic detection for mocking."""
        topics = []
        content_lower = content.lower()
        
        if "meeting" in content_lower or "agenda" in content_lower:
            topics.append("meetings")
        if "project" in content_lower or "milestone" in content_lower:
            topics.append("project management")
        if "research" in content_lower or "study" in content_lower:
            topics.append("research")
        if "daily" in content_lower or "tasks" in content_lower:
            topics.append("daily planning")
            
        return topics or ["general"]
    
    async def benchmark_content_classification(self):
        """Benchmark content classification speed."""
        print("\n=== Content Classification ===")
        
        # Get sample notes
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(100, len(notes)))
        
        # Individual classification
        times = []
        confidence_scores = []
        
        for note in sample_notes:
            start = time.perf_counter()
            result = await self.organizer.classifier.classify_content(note)
            end = time.perf_counter()
            
            times.append(end - start)
            confidence_scores.append(result.score)
        
        self.results["classification"] = times
        
        print(f"Content classification per note: {statistics.mean(times)*1000:.3f}ms (±{statistics.stdev(times)*1000:.3f}ms)")
        print(f"  Average confidence: {statistics.mean(confidence_scores):.3f}")
        
        # Batch classification
        start = time.perf_counter()
        
        batch_results = []
        for note in notes[:500]:  # First 500
            result = await self.organizer.classifier.classify_content(note)
            batch_results.append(result)
        
        end = time.perf_counter()
        
        print(f"Batch classification (500 notes): {end - start:.3f}s")
        print(f"  Average per note: {(end - start) / 500 * 1000:.3f}ms")
        
        # Classification by confidence level
        high_conf = sum(1 for r in batch_results if r.confidence == ClassificationConfidence.HIGH)
        med_conf = sum(1 for r in batch_results if r.confidence == ClassificationConfidence.MEDIUM)
        low_conf = sum(1 for r in batch_results if r.confidence == ClassificationConfidence.LOW)
        
        print(f"  High confidence: {high_conf} ({high_conf/len(batch_results)*100:.1f}%)")
        print(f"  Medium confidence: {med_conf} ({med_conf/len(batch_results)*100:.1f}%)")
        print(f"  Low confidence: {low_conf} ({low_conf/len(batch_results)*100:.1f}%)")
    
    async def benchmark_rule_evaluation(self):
        """Benchmark rule engine performance."""
        print("\n=== Rule Evaluation ===")
        
        # Add custom rules
        custom_rules = [
            OrganizationRule(
                name=f"custom_rule_{i}",
                conditions={
                    "filename_pattern": f".*pattern{i}.*",
                    "has_tag": f"tag{i}",
                    "min_word_count": 50,
                },
                action=OrganizationAction.MOVE,
                target_pattern=f"Custom/{i}/{{filename}}",
                priority=5 + i
            )
            for i in range(10)
        ]
        
        for rule in custom_rules:
            self.organizer.rule_engine.add_custom_rule(rule)
        
        # Benchmark rule evaluation
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(100, len(notes)))
        
        times = []
        matches_found = 0
        
        for note in sample_notes:
            features = await self.organizer.classifier.extract_features(note)
            
            start = time.perf_counter()
            results = await self.organizer.rule_engine.evaluate_rules(note, features)
            end = time.perf_counter()
            
            times.append(end - start)
            matches_found += len(results)
        
        self.results["rule_evaluation"] = times
        
        print(f"Rule evaluation per note: {statistics.mean(times)*1000:.3f}ms")
        print(f"  Total rules: {len(self.organizer.rule_engine.get_all_rules())}")
        print(f"  Average matches per note: {matches_found / len(sample_notes):.1f}")
        
        # Test with many rules
        print("\nTesting with many rules...")
        
        # Add more rules
        for i in range(50):
            rule = OrganizationRule(
                name=f"bulk_rule_{i}",
                conditions={"filename_pattern": f".*test{i}.*"},
                action=OrganizationAction.MOVE,
                target_pattern=f"Bulk/{i}/{{filename}}",
                priority=1
            )
            self.organizer.rule_engine.add_custom_rule(rule)
        
        # Re-benchmark
        start = time.perf_counter()
        
        for note in sample_notes[:50]:
            features = await self.organizer.classifier.extract_features(note)
            await self.organizer.rule_engine.evaluate_rules(note, features)
        
        end = time.perf_counter()
        
        many_rules_time = (end - start) / 50 * 1000
        print(f"  With {len(self.organizer.rule_engine.get_all_rules())} rules: {many_rules_time:.3f}ms per note")
    
    async def benchmark_directory_routing(self):
        """Benchmark directory routing operations."""
        print("\n=== Directory Routing ===")
        
        # Test path validation
        test_paths = [
            Path("Projects/Alpha/note.md"),
            Path("Daily Notes/2024/01/2024-01-15.md"),
            Path("Research/ML/Papers/paper1.md"),
            Path("../outside/bad.md"),
            Path("/absolute/path.md"),
        ]
        
        validation_times = []
        
        for path in test_paths * 20:  # Test multiple times
            start = time.perf_counter()
            is_valid, reason = await self.organizer.router._validate_target_path(path)
            end = time.perf_counter()
            
            validation_times.append(end - start)
        
        print(f"Path validation: {statistics.mean(validation_times)*1000:.3f}ms per path")
        
        # Test conflict resolution
        existing_files = list(self.vault_path.glob("**/*.md"))[:50]
        
        conflict_times = []
        
        for file in existing_files:
            relative_path = file.relative_to(self.vault_path)
            
            start = time.perf_counter()
            resolved = await self.organizer.router._resolve_naming_conflicts(relative_path)
            end = time.perf_counter()
            
            conflict_times.append(end - start)
        
        print(f"Conflict resolution: {statistics.mean(conflict_times)*1000:.3f}ms per file")
        
        # Test complete routing
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(50, len(notes)))
        
        routing_times = []
        
        for note in sample_notes:
            classification = await self.organizer.classifier.classify_content(note)
            
            start = time.perf_counter()
            routed_path = await self.organizer.router.route_file(note, classification)
            end = time.perf_counter()
            
            routing_times.append(end - start)
        
        self.results["routing"] = routing_times
        
        print(f"Complete routing: {statistics.mean(routing_times)*1000:.3f}ms per note")
    
    async def benchmark_batch_organization(self):
        """Benchmark batch organization operations."""
        print("\n=== Batch Organization ===")
        
        # Organize subset of vault
        print("Testing incremental organization...")
        
        start = time.perf_counter()
        results = await self.organizer.organize_vault(
            dry_run=True,
            max_files=100
        )
        end = time.perf_counter()
        
        subset_time = end - start
        print(f"Organize 100 files: {subset_time:.3f}s")
        print(f"  Processed: {results['processed']}")
        print(f"  Would organize: {results['organized']}")
        print(f"  Average per file: {subset_time / results['processed'] * 1000:.3f}ms")
        
        # Test with different confidence thresholds
        print("\nTesting confidence thresholds...")
        
        thresholds = [0.5, 0.7, 0.9]
        threshold_results = []
        
        for threshold in thresholds:
            self.config.auto_organization["confidence_threshold"] = threshold
            
            start = time.perf_counter()
            results = await self.organizer.organize_vault(
                dry_run=True,
                max_files=200
            )
            end = time.perf_counter()
            
            threshold_results.append({
                "threshold": threshold,
                "time": end - start,
                "organized": results["organized"],
                "processed": results["processed"],
            })
        
        for result in threshold_results:
            print(f"  Threshold {result['threshold']}: {result['organized']}/{result['processed']} files in {result['time']:.3f}s")
    
    async def benchmark_learning_system(self):
        """Benchmark learning and feedback system."""
        print("\n=== Learning System ===")
        
        from obsidian_librarian.services.auto_organizer import UserFeedback
        
        # Generate feedback history
        feedback_items = []
        
        for i in range(100):
            feedback = UserFeedback(
                original_path=Path(f"Unsorted/note_{i}.md"),
                suggested_path=Path(f"Projects/note_{i}.md"),
                actual_path=Path(f"Research/note_{i}.md") if i % 3 == 0 else Path(f"Projects/note_{i}.md"),
                accepted=i % 3 != 0,
                timestamp=datetime.utcnow() - timedelta(hours=i),
                feedback_type="correction" if i % 3 == 0 else "approval",
                confidence_score=random.uniform(0.5, 0.95)
            )
            feedback_items.append(feedback)
        
        # Benchmark feedback recording
        record_times = []
        
        for feedback in feedback_items[:50]:
            start = time.perf_counter()
            await self.organizer.learner.record_feedback(feedback)
            end = time.perf_counter()
            
            record_times.append(end - start)
        
        print(f"Feedback recording: {statistics.mean(record_times)*1000:.3f}ms per item")
        
        # Benchmark pattern learning
        start = time.perf_counter()
        
        # Trigger pattern learning
        for i in range(10):
            pattern = await self.organizer.learner.get_learned_pattern(f"similar-note-{i}.md")
        
        end = time.perf_counter()
        
        print(f"Pattern matching (10 queries): {end - start:.3f}s")
        
        # Benchmark statistics generation
        start = time.perf_counter()
        stats = await self.organizer.learner.get_statistics()
        end = time.perf_counter()
        
        print(f"Statistics generation: {(end - start)*1000:.3f}ms")
        print(f"  Total feedback: {stats['total_feedback']}")
        print(f"  Approval rate: {stats['approval_rate']:.1%}")
    
    async def benchmark_concurrent_operations(self):
        """Benchmark concurrent organization operations."""
        print("\n=== Concurrent Operations ===")
        
        # Get notes for testing
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(200, len(notes)))
        
        # Concurrent classification
        start = time.perf_counter()
        
        tasks = [
            self.organizer.classifier.classify_content(note)
            for note in sample_notes
        ]
        results = await asyncio.gather(*tasks)
        
        end = time.perf_counter()
        
        concurrent_time = end - start
        print(f"Concurrent classification (200 notes): {concurrent_time:.3f}s")
        print(f"  Average per note: {concurrent_time / 200 * 1000:.3f}ms")
        
        # Mixed concurrent operations
        start = time.perf_counter()
        
        mixed_tasks = []
        for i in range(100):
            if i % 4 == 0:
                # Classification
                note = random.choice(sample_notes)
                mixed_tasks.append(self.organizer.classifier.classify_content(note))
            elif i % 4 == 1:
                # Rule evaluation
                note = random.choice(sample_notes)
                features = await self.organizer.classifier.extract_features(note)
                mixed_tasks.append(self.organizer.rule_engine.evaluate_rules(note, features))
            elif i % 4 == 2:
                # Statistics
                mixed_tasks.append(self.organizer.get_statistics())
            else:
                # Structure analysis
                mixed_tasks.append(self.organizer.analyze_vault_structure())
        
        await asyncio.gather(*mixed_tasks, return_exceptions=True)
        
        end = time.perf_counter()
        
        print(f"Mixed concurrent operations (100 tasks): {end - start:.3f}s")
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory usage."""
        print("\n=== Memory Efficiency ===")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory before operations
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run full vault organization
        await self.organizer.organize_vault(dry_run=True)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage:")
        print(f"  Before organization: {mem_before:.1f} MB")
        print(f"  After organization: {mem_after:.1f} MB")
        print(f"  Increase: {mem_after - mem_before:.1f} MB")
        
        # Test with file watching
        print("\nFile watcher memory test:")
        
        # Start file watcher
        await self.organizer.start()
        
        mem_watching = process.memory_info().rss / 1024 / 1024  # MB
        
        # Stop file watcher
        await self.organizer.stop()
        
        mem_stopped = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  With file watcher: {mem_watching:.1f} MB")
        print(f"  After stopping: {mem_stopped:.1f} MB")
    
    async def run(self):
        """Run all benchmarks."""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"Organization Operations Benchmark - {self.num_notes} notes")
        print(separator)
        
        await self.setup()
        
        await self.benchmark_content_classification()
        await self.benchmark_rule_evaluation()
        await self.benchmark_directory_routing()
        await self.benchmark_batch_organization()
        await self.benchmark_learning_system()
        await self.benchmark_concurrent_operations()
        await self.benchmark_memory_efficiency()
        
        print(f"\n{separator}")
        print("Benchmark Summary")
        print(separator)
        
        # Summary statistics
        for operation, times in self.results.items():
            if times:
                avg_time = statistics.mean(times)
                if len(times) > 1:
                    std_time = statistics.stdev(times)
                    print(f"{operation}: {avg_time*1000:.3f}ms (±{std_time*1000:.3f}ms)")
                else:
                    print(f"{operation}: {avg_time:.3f}s")
        
        # Performance insights
        print("\nPerformance Insights:")
        
        if "classification" in self.results:
            class_times = self.results["classification"]
            fast_class = sum(1 for t in class_times if t < 0.01)  # < 10ms
            print(f"  Fast classifications (<10ms): {fast_class}/{len(class_times)} ({fast_class/len(class_times)*100:.1f}%)")
        
        if "routing" in self.results:
            route_times = self.results["routing"]
            fast_route = sum(1 for t in route_times if t < 0.005)  # < 5ms
            print(f"  Fast routing (<5ms): {fast_route}/{len(route_times)} ({fast_route/len(route_times)*100:.1f}%)")
        
        # Cleanup
        await self.vault.close()
        
        return self.results


async def main():
    """Run benchmarks with different vault sizes."""
    sizes = [100, 500, 1000]
    
    all_results = {}
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Running benchmark with {size} notes...")
        print(f"{'='*60}")
        
        benchmark = OrganizationBenchmark(num_notes=size)
        results = await benchmark.run()
        all_results[size] = results
        
        # Small delay between runs
        await asyncio.sleep(1)
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print("Comparative Analysis")
    print(f"{'='*60}")
    
    operations = ["classification", "routing"]
    
    for op in operations:
        print(f"\n{op.title()} Performance:")
        for size, results in all_results.items():
            if op in results and results[op]:
                avg_time = statistics.mean(results[op]) * 1000
                print(f"  {size} notes: {avg_time:.3f}ms per note")


if __name__ == "__main__":
    asyncio.run(main())