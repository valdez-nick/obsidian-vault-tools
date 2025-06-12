/*!
Benchmarks for file operations.
*/

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use librarian_core::{file_ops::FileOps, vault::{Vault, VaultConfig}};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::runtime::Runtime;

fn create_test_vault(num_files: usize) -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let vault_path = temp_dir.path().to_path_buf();

    // Create .obsidian directory
    std::fs::create_dir_all(vault_path.join(".obsidian")).unwrap();

    // Create test files
    for i in 0..num_files {
        let content = format!(
            r#"---
title: Test Note {}
tags: [test, benchmark]
created: 2024-01-01T00:00:00Z
---

# Test Note {}

This is a test note for benchmarking purposes. It contains some **markdown** formatting,
[links](http://example.com), and multiple paragraphs to simulate real content.

## Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua.

- [ ] Task 1 #todo
- [x] Completed task #done
- [ ] Another task #urgent

## Section 2

[[Another Note]] is referenced here, along with #tags and more content.

> This is a blockquote with some important information.

```rust
fn main() {{
    println!("Hello, world!");
}}
```

More content here to make the file reasonably sized for testing.
"#,
            i, i
        );

        std::fs::write(vault_path.join(format!("note_{:04}.md", i)), content).unwrap();
    }

    (temp_dir, vault_path)
}

fn bench_file_scanning(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_scanning");
    
    for size in [10, 100, 1000].iter() {
        let (_temp_dir, vault_path) = create_test_vault(*size);
        let file_ops = FileOps::new();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("scan_vault", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(file_ops.scan_vault(&vault_path).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_note_reading(c: &mut Criterion) {
    let mut group = c.benchmark_group("note_reading");
    let rt = Runtime::new().unwrap();
    
    // Create test files of different sizes
    let sizes = [
        ("small", 1000),   // ~1KB
        ("medium", 10000), // ~10KB
        ("large", 100000), // ~100KB
    ];

    for (size_name, content_size) in sizes.iter() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.md");
        
        let content = format!(
            "---\ntitle: Test\n---\n# Test\n\n{}",
            "Lorem ipsum dolor sit amet. ".repeat(*content_size / 30)
        );
        
        std::fs::write(&file_path, &content).unwrap();
        
        let file_ops = FileOps::new();
        
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("read_note_async", size_name),
            size_name,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    black_box(file_ops.read_note_async(&file_path).await.unwrap());
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("read_note_sync", size_name),
            size_name,
            |b, _| {
                b.iter(|| {
                    black_box(file_ops.read_note_sync(&file_path).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_vault_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_operations");
    let rt = Runtime::new().unwrap();
    
    for size in [100, 500, 1000].iter() {
        let (_temp_dir, vault_path) = create_test_vault(*size);
        let config = VaultConfig {
            path: vault_path,
            ..Default::default()
        };
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("vault_initialization", size),
            size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let vault = Vault::new(config.clone()).unwrap();
                    black_box(vault.initialize().await.unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_markdown_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("markdown_parsing");
    
    let test_content = r#"---
title: Complex Test Note
tags: [test, benchmark, complex]
initiative: test-project
product: benchmarks
priority: high
---

# Complex Test Note

This is a comprehensive test note with various markdown elements.

## Links and References

Here are some [[Internal Links]] and [[Complex Link|With Aliases]].
Also external [links](https://example.com) and more [[references]].

## Tasks and TODOs

- [x] Completed task #done #project/alpha
- [ ] Pending task #todo #urgent
- [ ] Another task with #multiple #tags
- [ ] Task with due date 📅 2024-01-01

## Tags and Content

This note has #inline-tags and #nested/tags/structure.
More #tags scattered throughout the content.

## Code and Formatting

```rust
fn example() {
    println!("Code block");
}
```

**Bold text**, *italic text*, and `inline code`.

> Important blockquote with information.

### Lists

1. Numbered list item
2. Another item
3. Third item

- Bullet point
- Another bullet
- Nested items
  - Sub-item
  - Another sub-item

## More Content

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
exercitation ullamco laboris.

### Additional Sections

More content with [[additional]] [[links]] and #more #tags for comprehensive testing.
"#;

    let parser = librarian_core::markdown::MarkdownParser::new();
    
    group.throughput(Throughput::Bytes(test_content.len() as u64));
    group.bench_function("parse_complex_note", |b| {
        b.iter(|| {
            black_box(parser.parse_note(test_content).unwrap());
        });
    });
    
    group.finish();
}

fn bench_content_hashing(c: &mut Criterion) {
    let mut group = c.benchmark_group("content_hashing");
    
    let sizes = [
        ("1kb", 1024),
        ("10kb", 10 * 1024),
        ("100kb", 100 * 1024),
        ("1mb", 1024 * 1024),
    ];

    for (size_name, size_bytes) in sizes.iter() {
        let content = "x".repeat(*size_bytes);
        
        group.throughput(Throughput::Bytes(*size_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("blake3_hash", size_name),
            &content,
            |b, content| {
                b.iter(|| {
                    black_box(librarian_core::Note::calculate_hash(content));
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_file_scanning,
    bench_note_reading,
    bench_vault_operations,
    bench_markdown_parsing,
    bench_content_hashing
);
criterion_main!(benches);