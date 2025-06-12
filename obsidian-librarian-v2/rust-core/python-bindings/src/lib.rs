/*!
Python bindings for Obsidian Librarian Rust core.
*/

use librarian_core::{
    file_ops::FileOps as RustFileOps,
    note::{Note as RustNote, NoteId as RustNoteId, NoteMetadata as RustNoteMetadata},
    vault::{Vault as RustVault, VaultConfig as RustVaultConfig, VaultStats as RustVaultStats},
    watcher::{VaultWatcher as RustVaultWatcher, VaultEvent as RustVaultEvent, WatcherConfig as RustWatcherConfig},
};
use pyo3::prelude::*;
use pyo3_asyncio::tokio::future_into_py;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Python wrapper for VaultConfig
#[pyclass]
#[derive(Clone)]
pub struct VaultConfig {
    inner: RustVaultConfig,
}

#[pymethods]
impl VaultConfig {
    #[new]
    fn new(
        path: String,
        exclude_patterns: Option<Vec<String>>,
        include_patterns: Option<Vec<String>>,
        max_file_size: Option<u64>,
        enable_cache: Option<bool>,
        cache_size_limit: Option<usize>,
    ) -> PyResult<Self> {
        let mut config = RustVaultConfig {
            path: PathBuf::from(path),
            ..Default::default()
        };

        if let Some(patterns) = exclude_patterns {
            config.exclude_patterns = patterns;
        }
        if let Some(patterns) = include_patterns {
            config.include_patterns = patterns;
        }
        if let Some(size) = max_file_size {
            config.max_file_size = size;
        }
        if let Some(cache) = enable_cache {
            config.enable_cache = cache;
        }
        if let Some(limit) = cache_size_limit {
            config.cache_size_limit = limit;
        }

        Ok(Self { inner: config })
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.to_string_lossy().to_string()
    }

    #[getter]
    fn exclude_patterns(&self) -> Vec<String> {
        self.inner.exclude_patterns.clone()
    }

    #[getter]
    fn include_patterns(&self) -> Vec<String> {
        self.inner.include_patterns.clone()
    }
}

/// Python wrapper for Note
#[pyclass]
#[derive(Clone)]
pub struct Note {
    inner: RustNote,
}

#[pymethods]
impl Note {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.as_str().to_string()
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.to_string_lossy().to_string()
    }

    #[getter]
    fn content(&self) -> String {
        self.inner.content.clone()
    }

    #[getter]
    fn content_hash(&self) -> String {
        self.inner.content_hash.clone()
    }

    #[getter]
    fn word_count(&self) -> usize {
        self.inner.word_count
    }

    #[getter]
    fn file_size(&self) -> u64 {
        self.inner.file_size
    }

    #[getter]
    fn title(&self) -> Option<String> {
        self.inner.metadata.title.clone()
    }

    #[getter]
    fn tags(&self) -> Vec<String> {
        self.inner.metadata.tags.iter().cloned().collect()
    }

    #[getter]
    fn links(&self) -> Vec<HashMap<String, Option<String>>> {
        self.inner
            .links
            .iter()
            .map(|link| {
                let mut map = HashMap::new();
                map.insert("target".to_string(), Some(link.target.clone()));
                map.insert("alias".to_string(), link.alias.clone());
                map.insert("text".to_string(), Some(link.text.clone()));
                map
            })
            .collect()
    }

    #[getter]
    fn tasks(&self) -> Vec<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            self.inner
                .tasks
                .iter()
                .map(|task| {
                    let mut map = HashMap::new();
                    map.insert("text".to_string(), task.text.clone().into_py(py));
                    map.insert("completed".to_string(), task.completed.into_py(py));
                    map.insert("line".to_string(), task.line.into_py(py));
                    map.insert("position".to_string(), task.position.into_py(py));
                    map.insert(
                        "tags".to_string(),
                        task.tags.iter().cloned().collect::<Vec<_>>().into_py(py),
                    );
                    map
                })
                .collect()
        })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    #[classmethod]
    fn from_json(_cls: &PyType, json: &str) -> PyResult<Self> {
        let inner = RustNote::from_json(json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// Python wrapper for VaultStats
#[pyclass]
#[derive(Clone)]
pub struct VaultStats {
    inner: RustVaultStats,
}

#[pymethods]
impl VaultStats {
    #[getter]
    fn note_count(&self) -> usize {
        self.inner.note_count
    }

    #[getter]
    fn folder_count(&self) -> usize {
        self.inner.folder_count
    }

    #[getter]
    fn total_size(&self) -> u64 {
        self.inner.total_size
    }
}

/// Python wrapper for Vault
#[pyclass]
pub struct Vault {
    inner: RustVault,
    runtime: tokio::runtime::Handle,
}

#[pymethods]
impl Vault {
    #[new]
    fn new(config: VaultConfig) -> PyResult<Self> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No tokio runtime found. Please ensure you're running in an async context."
            ))?;

        let inner = RustVault::new(config.inner)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self { inner, runtime })
    }

    fn initialize<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        future_into_py(py, async move {
            vault
                .initialize()
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    fn read_note<'p>(&self, py: Python<'p>, path: String) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        let path = PathBuf::from(path);
        future_into_py(py, async move {
            let note = vault
                .read_note(path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            Ok(Note { inner: note })
        })
    }

    fn write_note<'p>(&self, py: Python<'p>, note: Note) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        future_into_py(py, async move {
            vault
                .write_note(&note.inner)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
        })
    }

    fn create_note<'p>(
        &self,
        py: Python<'p>,
        path: String,
        content: String,
    ) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        let path = PathBuf::from(path);
        future_into_py(py, async move {
            let note = vault
                .create_note(path, content)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            Ok(Note { inner: note })
        })
    }

    fn delete_note<'p>(&self, py: Python<'p>, path: String) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        let path = PathBuf::from(path);
        future_into_py(py, async move {
            let trash_path = vault
                .delete_note(path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            Ok(trash_path.to_string_lossy().to_string())
        })
    }

    fn get_all_notes<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        future_into_py(py, async move {
            let notes = vault
                .get_all_notes()
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(notes
                .into_iter()
                .map(|note| Note { inner: note })
                .collect::<Vec<_>>())
        })
    }

    fn find_notes_by_tag<'p>(&self, py: Python<'p>, tag: String) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        future_into_py(py, async move {
            let notes = vault
                .find_notes_by_tag(&tag)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(notes
                .into_iter()
                .map(|note| Note { inner: note })
                .collect::<Vec<_>>())
        })
    }

    fn get_stats<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        future_into_py(py, async move {
            let stats = vault.get_stats().await;
            Ok(VaultStats { inner: stats })
        })
    }

    fn clear_cache<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let vault = self.inner.clone();
        future_into_py(py, async move {
            vault.clear_cache().await;
            Ok(())
        })
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path().to_string_lossy().to_string()
    }
}

/// Python wrapper for file operations
#[pyclass]
pub struct FileOps {
    inner: RustFileOps,
}

#[pymethods]
impl FileOps {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustFileOps::new(),
        }
    }

    fn read_note<'p>(&self, py: Python<'p>, path: String) -> PyResult<&'p PyAny> {
        let file_ops = self.inner.clone();
        let path = PathBuf::from(path);
        future_into_py(py, async move {
            let note = file_ops
                .read_note_async(path)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            Ok(Note { inner: note })
        })
    }

    fn scan_vault(&self, vault_path: String) -> PyResult<Vec<String>> {
        let paths = self
            .inner
            .scan_vault(PathBuf::from(vault_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(paths
            .into_iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect())
    }

    fn calculate_stats(&self, vault_path: String) -> PyResult<VaultStats> {
        let stats = self
            .inner
            .calculate_stats(PathBuf::from(vault_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(VaultStats { inner: stats })
    }
}

/// Python wrapper for VaultEvent
#[pyclass]
#[derive(Clone)]
pub struct VaultEvent {
    inner: RustVaultEvent,
}

#[pymethods]
impl VaultEvent {
    #[getter]
    fn event_type(&self) -> String {
        match &self.inner {
            RustVaultEvent::Created { .. } => "created".to_string(),
            RustVaultEvent::Modified { .. } => "modified".to_string(),
            RustVaultEvent::Deleted { .. } => "deleted".to_string(),
            RustVaultEvent::Moved { .. } => "moved".to_string(),
        }
    }

    #[getter]
    fn path(&self) -> String {
        match &self.inner {
            RustVaultEvent::Created { path } 
            | RustVaultEvent::Modified { path } 
            | RustVaultEvent::Deleted { path } => path.to_string_lossy().to_string(),
            RustVaultEvent::Moved { to, .. } => to.to_string_lossy().to_string(),
        }
    }

    #[getter]
    fn old_path(&self) -> Option<String> {
        match &self.inner {
            RustVaultEvent::Moved { from, .. } => Some(from.to_string_lossy().to_string()),
            _ => None,
        }
    }
}

/// Python wrapper for WatcherConfig
#[pyclass]
#[derive(Clone)]
pub struct WatcherConfig {
    inner: RustWatcherConfig,
}

#[pymethods]
impl WatcherConfig {
    #[new]
    fn new(
        debounce_duration_ms: Option<u64>,
        buffer_size: Option<usize>,
        exclude_patterns: Option<Vec<String>>,
    ) -> Self {
        let mut config = RustWatcherConfig::default();

        if let Some(duration) = debounce_duration_ms {
            config.debounce_duration = Duration::from_millis(duration);
        }
        if let Some(size) = buffer_size {
            config.buffer_size = size;
        }
        if let Some(patterns) = exclude_patterns {
            config.exclude_patterns = patterns;
        }

        Self { inner: config }
    }
}

/// Module initialization
#[pymodule]
fn librarian_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<VaultConfig>()?;
    m.add_class::<Note>()?;
    m.add_class::<VaultStats>()?;
    m.add_class::<Vault>()?;
    m.add_class::<FileOps>()?;
    m.add_class::<VaultEvent>()?;
    m.add_class::<WatcherConfig>()?;
    Ok(())
}