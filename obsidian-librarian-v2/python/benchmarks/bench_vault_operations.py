"""
Performance benchmarks for vault operations.
"""

import asyncio
import time
import statistics
from pathlib import Path
import tempfile
import random
import string
from typing import List, Dict, Any

from obsidian_librarian import Vault
from obsidian_librarian.models import VaultConfig


class VaultBenchmark:
    """Benchmark vault operations."""
    
    def __init__(self, num_notes: int = 1000):
        self.num_notes = num_notes
        self.results: Dict[str, List[float]] = {}
    
    async def setup(self):
        """Set up test vault with sample data."""
        self.tmpdir = tempfile.mkdtemp()
        self.vault_path = Path(self.tmpdir) / "bench_vault"
        self.vault_path.mkdir()
        
        # Create vault structure
        (self.vault_path / ".obsidian").mkdir()
        
        # Generate sample notes
        print(f"Creating {self.num_notes} sample notes...")
        for i in range(self.num_notes):
            content = self._generate_note_content(i)
            (self.vault_path / f"note_{i:04d}.md").write_text(content)
        
        # Initialize vault
        config = VaultConfig(
            cache_size=10000,
            enable_file_watching=False,
        )
        self.vault = Vault(self.vault_path, config)
        await self.vault.initialize()
    
    def _generate_note_content(self, index: int) -> str:
        """Generate random note content."""
        tags = random.sample(["programming", "python", "rust", "ml", "ai", "data"], k=2)
        links = [f"[[note_{random.randint(0, self.num_notes-1):04d}]]" for _ in range(random.randint(0, 5))]
        
        return f"""---
title: Note {index}
tags: {tags}
created: 2024-01-{(index % 30) + 1:02d}
---

# Note {index}

This is a sample note for benchmarking purposes.

## Content

{''.join(random.choices(string.ascii_letters + string.digits + ' \n', k=random.randint(500, 2000)))}

## Links

{' '.join(links)}

## Tasks

- [ ] Task 1
- [x] Task 2 completed
- [ ] Task 3 #todo
"""
    
    async def benchmark_read_operations(self):
        """Benchmark read operations."""
        print("\n=== Read Operations ===")
        
        # Get all notes
        times = []
        for _ in range(5):
            start = time.perf_counter()
            notes = await self.vault.get_all_notes()
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["get_all_notes"] = times
        print(f"Get all notes ({len(notes)}): {statistics.mean(times):.3f}s (±{statistics.stdev(times):.3f}s)")
        
        # Get individual notes
        note_ids = [n.id for n in notes[:100]]
        times = []
        for note_id in note_ids:
            start = time.perf_counter()
            await self.vault.get_note(note_id)
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["get_single_note"] = times
        print(f"Get single note (100 samples): {statistics.mean(times)*1000:.3f}ms (±{statistics.stdev(times)*1000:.3f}ms)")
    
    async def benchmark_search_operations(self):
        """Benchmark search operations."""
        print("\n=== Search Operations ===")
        
        search_queries = [
            "python",
            "machine learning",
            "task",
            "note",
            "completed",
        ]
        
        # Full-text search
        times = []
        for query in search_queries:
            start = time.perf_counter()
            results = await self.vault.search_notes(query, limit=50)
            end = time.perf_counter()
            times.append(end - start)
            print(f"  Search '{query}': {len(results)} results in {(end-start)*1000:.1f}ms")
        
        self.results["search_notes"] = times
        print(f"Average search time: {statistics.mean(times)*1000:.3f}ms")
        
        # Tag search
        times = []
        for tag in ["python", "ml", "data"]:
            start = time.perf_counter()
            results = await self.vault.get_notes_by_tag(tag)
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["search_by_tag"] = times
        print(f"Search by tag: {statistics.mean(times)*1000:.3f}ms")
    
    async def benchmark_link_operations(self):
        """Benchmark link-related operations."""
        print("\n=== Link Operations ===")
        
        # Get sample notes
        notes = await self.vault.get_all_notes()
        sample_notes = random.sample(notes, min(50, len(notes)))
        
        # Get linked notes
        times = []
        for note in sample_notes:
            start = time.perf_counter()
            linked = await self.vault.get_linked_notes(note.id)
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["get_linked_notes"] = times
        print(f"Get linked notes: {statistics.mean(times)*1000:.3f}ms")
        
        # Get backlinks
        times = []
        for note in sample_notes:
            start = time.perf_counter()
            backlinks = await self.vault.get_backlinks(note.id)
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["get_backlinks"] = times
        print(f"Get backlinks: {statistics.mean(times)*1000:.3f}ms")
        
        # Find orphaned notes
        start = time.perf_counter()
        orphans = await self.vault.get_orphaned_notes()
        end = time.perf_counter()
        print(f"Find orphaned notes: {len(orphans)} found in {(end-start)*1000:.1f}ms")
    
    async def benchmark_write_operations(self):
        """Benchmark write operations."""
        print("\n=== Write Operations ===")
        
        # Create notes
        times = []
        for i in range(10):
            content = self._generate_note_content(self.num_notes + i)
            start = time.perf_counter()
            note_id = await self.vault.create_note(
                Path(f"bench_create_{i}.md"),
                content,
                {"benchmark": True}
            )
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["create_note"] = times
        print(f"Create note: {statistics.mean(times)*1000:.3f}ms")
        
        # Update notes
        times = []
        for i in range(10):
            note_id = f"bench_create_{i}.md"
            start = time.perf_counter()
            await self.vault.update_note(
                note_id,
                f"# Updated\n\nNew content {i}",
                {"updated": True}
            )
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["update_note"] = times
        print(f"Update note: {statistics.mean(times)*1000:.3f}ms")
        
        # Delete notes
        times = []
        for i in range(10):
            note_id = f"bench_create_{i}.md"
            start = time.perf_counter()
            await self.vault.delete_note(note_id)
            end = time.perf_counter()
            times.append(end - start)
        
        self.results["delete_note"] = times
        print(f"Delete note: {statistics.mean(times)*1000:.3f}ms")
    
    async def benchmark_concurrent_operations(self):
        """Benchmark concurrent operations."""
        print("\n=== Concurrent Operations ===")
        
        # Concurrent reads
        start = time.perf_counter()
        tasks = []
        for _ in range(100):
            note_id = f"note_{random.randint(0, self.num_notes-1):04d}.md"
            tasks.append(self.vault.get_note(note_id))
        
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        print(f"100 concurrent reads: {(end-start)*1000:.1f}ms total")
        
        # Concurrent searches
        start = time.perf_counter()
        tasks = []
        for query in ["python", "rust", "ml", "task", "note"]:
            for _ in range(5):
                tasks.append(self.vault.search_notes(query, limit=10))
        
        results = await asyncio.gather(*tasks)
        end = time.perf_counter()
        print(f"25 concurrent searches: {(end-start)*1000:.1f}ms total")
        
        # Mixed operations
        start = time.perf_counter()
        tasks = []
        
        # Mix of reads, searches, and link operations
        for _ in range(20):
            tasks.append(self.vault.get_all_note_ids())
        
        for _ in range(20):
            note_id = f"note_{random.randint(0, self.num_notes-1):04d}.md"
            tasks.append(self.vault.get_note(note_id))
        
        for _ in range(20):
            tasks.append(self.vault.search_notes("test", limit=5))
        
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        print(f"60 mixed concurrent operations: {(end-start)*1000:.1f}ms total")
    
    async def benchmark_stats_operations(self):
        """Benchmark statistics operations."""
        print("\n=== Statistics Operations ===")
        
        times = []
        for _ in range(5):
            start = time.perf_counter()
            stats = await self.vault.get_stats()
            end = time.perf_counter()
            times.append(end - start)
        
        print(f"Get vault stats: {statistics.mean(times)*1000:.3f}ms")
        print(f"  Total notes: {stats.total_notes}")
        print(f"  Total words: {stats.total_words}")
        print(f"  Total links: {stats.total_links}")
        print(f"  Total tasks: {stats.total_tasks}")
    
    async def run(self):
        """Run all benchmarks."""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"Vault Performance Benchmark - {self.num_notes} notes")
        print(separator)
        
        await self.setup()
        
        await self.benchmark_read_operations()
        await self.benchmark_search_operations()
        await self.benchmark_link_operations()
        await self.benchmark_write_operations()
        await self.benchmark_concurrent_operations()
        await self.benchmark_stats_operations()
        
        separator = "=" * 60
        print(f"\n{separator}")
        print("Benchmark complete!")
        
        # Cleanup
        await self.vault.close()
        
        return self.results


async def main():
    """Run benchmarks with different vault sizes."""
    sizes = [100, 1000, 5000]
    
    for size in sizes:
        benchmark = VaultBenchmark(num_notes=size)
        await benchmark.run()
        
        # Small delay between runs
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())