"""
Performance benchmarks for tag management operations.

Benchmarks various tag operations including:
- Tag extraction and analysis
- Similarity detection  
- Hierarchy building
- Auto-tagging
- Bulk tag operations
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

from obsidian_librarian import Vault
from obsidian_librarian.models import VaultConfig, TagManagerConfig
from obsidian_librarian.services.tag_manager import TagManagerService
from obsidian_librarian.ai.embeddings import EmbeddingService


class TagBenchmark:
    """Benchmark tag management operations."""
    
    def __init__(self, num_notes: int = 1000, tags_per_note: int = 5):
        self.num_notes = num_notes
        self.tags_per_note = tags_per_note
        self.results: Dict[str, List[float]] = {}
        
        # Common tags for realistic distribution
        self.common_tags = [
            "project", "meeting", "todo", "idea", "research",
            "development", "design", "documentation", "review", "planning",
            "python", "javascript", "api", "database", "frontend",
            "backend", "testing", "deployment", "security", "performance"
        ]
        
        # Tag variations for similarity testing
        self.tag_variations = {
            "ml": ["ML", "machine-learning", "MachineLearning", "machine_learning"],
            "api": ["API", "Api", "apis", "APIs"],
            "todo": ["TODO", "To-Do", "to-do", "ToDo"],
            "project": ["Project", "PROJECT", "projects", "Projects"],
        }
    
    async def setup(self):
        """Set up test vault with tagged notes."""
        self.tmpdir = tempfile.mkdtemp()
        self.vault_path = Path(self.tmpdir) / "bench_vault"
        self.vault_path.mkdir()
        
        # Create vault structure
        (self.vault_path / ".obsidian").mkdir()
        directories = ["Projects", "Meetings", "Research", "Daily Notes"]
        for dir_name in directories:
            (self.vault_path / dir_name).mkdir()
        
        # Generate notes with tags
        print(f"Creating {self.num_notes} notes with tags...")
        self.note_tags = {}
        
        for i in range(self.num_notes):
            tags = self._generate_tags(i)
            self.note_tags[f"note_{i:04d}.md"] = tags
            
            content = self._generate_note_with_tags(i, tags)
            (self.vault_path / f"note_{i:04d}.md").write_text(content)
        
        # Initialize vault and services
        config = VaultConfig(
            cache_size=10000,
            enable_file_watching=False,
        )
        self.vault = Vault(self.vault_path, config)
        await self.vault.initialize()
        
        # Initialize tag manager
        tag_config = TagManagerConfig(
            fuzzy_similarity_threshold=0.8,
            semantic_similarity_threshold=0.7,
            enable_fuzzy_matching=True,
            enable_semantic_analysis=False,  # Disable for performance testing
            case_insensitive=True,
            min_usage_threshold=2,
            auto_tag_confidence_threshold=0.7,
            max_auto_tags_per_note=5,
        )
        
        # Mock embedding service for benchmarks
        mock_embedding_service = MockEmbeddingService()
        
        self.tag_manager = TagManagerService(
            vault=self.vault,
            config=tag_config,
            embedding_service=mock_embedding_service
        )
    
    def _generate_tags(self, index: int) -> List[str]:
        """Generate realistic tag distribution."""
        tags = []
        
        # Add some common tags (power law distribution)
        num_common = random.randint(1, 3)
        tags.extend(random.sample(self.common_tags, num_common))
        
        # Add variations (for similarity testing)
        if index % 10 == 0:
            variation_key = random.choice(list(self.tag_variations.keys()))
            tags.append(random.choice(self.tag_variations[variation_key]))
        
        # Add unique tags
        num_unique = self.tags_per_note - len(tags)
        for _ in range(num_unique):
            tags.append(f"tag-{random.randint(0, 100)}")
        
        # Add hierarchical tags occasionally
        if index % 5 == 0:
            parent = random.choice(["project", "research", "team"])
            child = random.choice(["alpha", "beta", "gamma"])
            tags.append(f"{parent}/{child}")
        
        return tags
    
    def _generate_note_with_tags(self, index: int, tags: List[str]) -> str:
        """Generate note content with tags."""
        # Mix of frontmatter and inline tags
        frontmatter_tags = tags[:len(tags)//2]
        inline_tags = tags[len(tags)//2:]
        
        content = f"""---
title: Note {index}
tags: {frontmatter_tags}
created: 2024-01-{(index % 30) + 1:02d}
---

# Note {index}

This is a benchmark note with various tags.

## Content

Some content here with inline tags: {' '.join(f'#{tag}' for tag in inline_tags)}

{''.join(random.choices(string.ascii_letters + ' \n', k=random.randint(200, 500)))}

## Related
- [[note_{random.randint(0, self.num_notes-1):04d}]]
- [[note_{random.randint(0, self.num_notes-1):04d}]]
"""
        return content
    
    async def benchmark_tag_extraction(self):
        """Benchmark tag extraction from notes."""
        print("\n=== Tag Extraction ===")
        
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(100, len(notes)))
        
        times = []
        for note in sample_notes:
            start = time.perf_counter()
            tags = await self.tag_manager.analyzer.extract_tags_from_note(note)
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["tag_extraction"] = times
        print(f"Tag extraction per note: {statistics.mean(times)*1000:.3f}ms (±{statistics.stdev(times)*1000:.3f}ms)")
        
        # Benchmark batch extraction
        start = time.perf_counter()
        all_tags = {}
        for note in notes[:500]:  # First 500 notes
            tags = await self.tag_manager.analyzer.extract_tags_from_note(note)
            all_tags[note.id] = tags
        end = time.perf_counter()
        
        print(f"Batch extraction (500 notes): {end - start:.3f}s")
        print(f"  Average per note: {(end - start) / 500 * 1000:.3f}ms")
    
    async def benchmark_tag_analysis(self):
        """Benchmark complete tag analysis."""
        print("\n=== Tag Analysis ===")
        
        # Full vault analysis
        start = time.perf_counter()
        analysis_result = await self.tag_manager.analyze_tags()
        end = time.perf_counter()
        
        analysis_time = end - start
        self.results["full_analysis"] = [analysis_time]
        
        print(f"Full tag analysis: {analysis_time:.3f}s")
        print(f"  Total tags found: {analysis_result.total_tags}")
        print(f"  Unique tags: {analysis_result.unique_tags}")
        print(f"  Tag clusters: {len(analysis_result.tag_clusters)}")
        print(f"  Suggested hierarchies: {len(analysis_result.suggested_hierarchies)}")
        
        # Analyze subsets
        subset_sizes = [100, 250, 500]
        subset_times = []
        
        for size in subset_sizes:
            notes = await self.vault.get_all_notes()
            subset = notes[:size]
            
            start = time.perf_counter()
            tag_info = await self.tag_manager.analyzer.analyze_tag_usage(subset)
            end = time.perf_counter()
            
            subset_time = end - start
            subset_times.append(subset_time)
            print(f"  Subset analysis ({size} notes): {subset_time:.3f}s")
        
        self.results["subset_analysis"] = subset_times
    
    async def benchmark_similarity_detection(self):
        """Benchmark tag similarity detection."""
        print("\n=== Similarity Detection ===")
        
        # Get all unique tags
        all_notes = await self.vault.get_all_notes()
        all_tags = set()
        for note in all_notes[:500]:  # Use first 500 notes
            tags = await self.tag_manager.analyzer.extract_tags_from_note(note)
            all_tags.update(tags)
        
        tag_list = list(all_tags)
        print(f"  Testing with {len(tag_list)} unique tags")
        
        # Benchmark pairwise similarity
        sample_size = min(50, len(tag_list))
        sample_tags = random.sample(tag_list, sample_size)
        
        times = []
        similarities_found = 0
        
        for i, tag1 in enumerate(sample_tags):
            for tag2 in sample_tags[i+1:]:
                start = time.perf_counter()
                similarity = await self.tag_manager.similarity_detector.calculate_similarity(tag1, tag2)
                end = time.perf_counter()
                
                times.append(end - start)
                if similarity and similarity.similarity_score > 0.8:
                    similarities_found += 1
        
        self.results["pairwise_similarity"] = times
        comparisons = len(times)
        print(f"Pairwise similarity ({comparisons} comparisons):")
        print(f"  Average time: {statistics.mean(times)*1000:.3f}ms")
        print(f"  Similar pairs found: {similarities_found}")
        
        # Benchmark batch similarity detection
        start = time.perf_counter()
        all_similarities = await self.tag_manager.find_similar_tags(tag_list)
        end = time.perf_counter()
        
        print(f"Batch similarity detection: {end - start:.3f}s")
        print(f"  Similarities found: {len(all_similarities)}")
        
        # Benchmark clustering
        start = time.perf_counter()
        clusters = await self.tag_manager.similarity_detector.cluster_similar_tags(tag_list)
        end = time.perf_counter()
        
        print(f"Tag clustering: {end - start:.3f}s")
        print(f"  Clusters formed: {len(clusters)}")
    
    async def benchmark_auto_tagging(self):
        """Benchmark auto-tagging suggestions."""
        print("\n=== Auto-Tagging ===")
        
        # Create some untagged notes
        untagged_notes = []
        for i in range(20):
            content = f"""# Untagged Note {i}

This note discusses machine learning and Python programming.
We're building an API for data processing.

## Tasks
- [ ] Implement algorithm
- [ ] Write tests
- [ ] Deploy to production
"""
            path = self.vault_path / f"untagged_{i}.md"
            path.write_text(content)
            
            note = await self.vault.get_note(f"untagged_{i}.md")
            if note:
                untagged_notes.append(note)
        
        # Get context tags
        tag_info = await self.tag_manager._get_tag_info_cached()
        
        # Benchmark individual suggestions
        times = []
        total_suggestions = 0
        
        for note in untagged_notes:
            start = time.perf_counter()
            suggestions = await self.tag_manager.suggest_tags(
                note.id,
                max_suggestions=5
            )
            end = time.perf_counter()
            
            times.append(end - start)
            total_suggestions += len(suggestions)
        
        self.results["auto_tag_suggestions"] = times
        print(f"Auto-tag suggestions per note: {statistics.mean(times)*1000:.3f}ms")
        print(f"  Average suggestions per note: {total_suggestions / len(untagged_notes):.1f}")
        
        # Benchmark batch auto-tagging
        start = time.perf_counter()
        
        tagged_count = 0
        async for progress in self.tag_manager.auto_tag_untagged_notes(
            confidence_threshold=0.7,
            dry_run=True
        ):
            if progress.get("type") == "progress":
                tagged_count += 1
        
        end = time.perf_counter()
        
        print(f"Batch auto-tagging ({len(untagged_notes)} notes): {end - start:.3f}s")
    
    async def benchmark_bulk_operations(self):
        """Benchmark bulk tag operations."""
        print("\n=== Bulk Operations ===")
        
        # Prepare merge operations
        merge_map = {
            "ML": "machine-learning",
            "ml": "machine-learning",
            "API": "api",
            "Todo": "todo",
            "TODO": "todo",
        }
        
        # Benchmark tag merging
        start = time.perf_counter()
        merge_results = await self.tag_manager.merge_tags(merge_map, dry_run=True)
        end = time.perf_counter()
        
        merge_time = end - start
        self.results["bulk_merge"] = [merge_time]
        
        affected_notes = sum(len(r.affected_notes) for r in merge_results if r.affected_notes)
        print(f"Tag merge operation: {merge_time:.3f}s")
        print(f"  Tags to merge: {len(merge_map)}")
        print(f"  Notes affected: {affected_notes}")
        
        # Benchmark bulk tag addition
        note_ids = [f"note_{i:04d}.md" for i in range(100)]
        new_tags = ["benchmark", "test"]
        
        start = time.perf_counter()
        add_results = await self.tag_manager.operations.add_tags_to_notes(
            note_ids,
            new_tags,
            dry_run=True
        )
        end = time.perf_counter()
        
        add_time = end - start
        print(f"Bulk tag addition (100 notes): {add_time:.3f}s")
        print(f"  Average per note: {add_time / 100 * 1000:.3f}ms")
        
        # Benchmark tag renaming
        start = time.perf_counter()
        rename_results = await self.tag_manager.operations.rename_tag(
            "project",
            "project-work",
            dry_run=True
        )
        end = time.perf_counter()
        
        rename_time = end - start
        affected = len([r for r in rename_results if r.success])
        print(f"Tag rename operation: {rename_time:.3f}s")
        print(f"  Notes affected: {affected}")
    
    async def benchmark_hierarchy_operations(self):
        """Benchmark tag hierarchy operations."""
        print("\n=== Hierarchy Operations ===")
        
        # Get tag info for hierarchy building
        tag_info = await self.tag_manager._get_tag_info_cached()
        
        # Benchmark hierarchy detection from paths
        start = time.perf_counter()
        path_hierarchies = await self.tag_manager.hierarchy_builder.build_from_paths(tag_info)
        end = time.perf_counter()
        
        path_time = end - start
        print(f"Path-based hierarchy detection: {path_time:.3f}s")
        print(f"  Hierarchies found: {len(path_hierarchies)}")
        
        # Benchmark co-occurrence analysis
        start = time.perf_counter()
        cooccurrence = await self.tag_manager.analyzer.get_tag_cooccurrence_matrix(tag_info)
        end = time.perf_counter()
        
        cooc_time = end - start
        print(f"Co-occurrence matrix generation: {cooc_time:.3f}s")
        print(f"  Matrix size: {len(cooccurrence)} pairs")
        
        # Benchmark hierarchy suggestions
        start = time.perf_counter()
        suggested_hierarchies = await self.tag_manager.hierarchy_builder.suggest_from_cooccurrence(
            tag_info,
            cooccurrence
        )
        end = time.perf_counter()
        
        suggest_time = end - start
        print(f"Hierarchy suggestions from co-occurrence: {suggest_time:.3f}s")
        print(f"  Suggestions generated: {len(suggested_hierarchies)}")
    
    async def benchmark_concurrent_operations(self):
        """Benchmark concurrent tag operations."""
        print("\n=== Concurrent Operations ===")
        
        # Concurrent tag extraction
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(200, len(notes)))
        
        start = time.perf_counter()
        tasks = [
            self.tag_manager.analyzer.extract_tags_from_note(note)
            for note in sample_notes
        ]
        results = await asyncio.gather(*tasks)
        end = time.perf_counter()
        
        concurrent_time = end - start
        print(f"Concurrent tag extraction (200 notes): {concurrent_time:.3f}s")
        print(f"  Average per note: {concurrent_time / 200 * 1000:.3f}ms")
        
        # Mixed concurrent operations
        start = time.perf_counter()
        mixed_tasks = []
        
        # Mix different operations
        for i in range(50):
            if i % 3 == 0:
                # Tag extraction
                note_id = f"note_{random.randint(0, self.num_notes-1):04d}.md"
                mixed_tasks.append(
                    self.tag_manager.suggest_tags(note_id, max_suggestions=3)
                )
            elif i % 3 == 1:
                # Statistics
                mixed_tasks.append(
                    self.tag_manager.get_tag_statistics()
                )
            else:
                # Similarity check
                tag1 = random.choice(self.common_tags)
                tag2 = random.choice(self.common_tags)
                mixed_tasks.append(
                    self.tag_manager.similarity_detector.calculate_similarity(tag1, tag2)
                )
        
        await asyncio.gather(*mixed_tasks, return_exceptions=True)
        end = time.perf_counter()
        
        print(f"Mixed concurrent operations (50 tasks): {end - start:.3f}s")
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory usage for large operations."""
        print("\n=== Memory Efficiency ===")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Memory before analysis
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run full analysis
        await self.tag_manager.analyze_tags()
        
        # Memory after analysis
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage:")
        print(f"  Before analysis: {mem_before:.1f} MB")
        print(f"  After analysis: {mem_after:.1f} MB")
        print(f"  Increase: {mem_after - mem_before:.1f} MB")
        
        # Test cache efficiency
        print("\nCache efficiency test:")
        
        # First call (builds cache)
        start = time.perf_counter()
        tag_info1 = await self.tag_manager._get_tag_info_cached()
        end = time.perf_counter()
        first_call = end - start
        
        # Second call (uses cache)
        start = time.perf_counter()
        tag_info2 = await self.tag_manager._get_tag_info_cached()
        end = time.perf_counter()
        second_call = end - start
        
        print(f"  First call (build cache): {first_call:.3f}s")
        print(f"  Second call (from cache): {second_call:.3f}s")
        print(f"  Speedup: {first_call / second_call:.1f}x")
    
    async def run(self):
        """Run all benchmarks."""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"Tag Operations Benchmark - {self.num_notes} notes")
        print(separator)
        
        await self.setup()
        
        await self.benchmark_tag_extraction()
        await self.benchmark_tag_analysis()
        await self.benchmark_similarity_detection()
        await self.benchmark_auto_tagging()
        await self.benchmark_bulk_operations()
        await self.benchmark_hierarchy_operations()
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
        
        # Cleanup
        await self.vault.close()
        
        return self.results


class MockEmbeddingService:
    """Mock embedding service for benchmarking."""
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate fake embedding based on text hash."""
        # Simple deterministic embedding
        hash_value = hash(text)
        embedding = []
        for i in range(10):  # 10-dimensional embedding
            value = ((hash_value >> (i * 4)) & 0xFF) / 255.0
            embedding.append(value)
        return embedding
    
    async def compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np
        
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


async def main():
    """Run benchmarks with different vault sizes."""
    sizes = [100, 500, 1000]
    
    for size in sizes:
        benchmark = TagBenchmark(num_notes=size, tags_per_note=5)
        await benchmark.run()
        
        # Small delay between runs
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())