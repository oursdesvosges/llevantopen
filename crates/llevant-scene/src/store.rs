//! A disk cache of compiled models, so that loading an input compiled
//! before takes a read ([`Model::read`]) instead of a compile (about 1.5 s
//! instead of 50 s for the reference model).
//!
//! An entry is keyed by the input's bytes, the [`Options`] and the engine
//! build ([`llevant::ENGINE_HASH`]), so any change to either misses. The
//! most recently used entries are kept within a budget (8 GB by default),
//! so small entries (tests) never push out large ones. A cache that cannot
//! be read or written is ignored: loading then compiles.
//!
//! The cache also remembers the last entry of each input path. When an
//! input has changed since, loading reads that entry and recompiles against
//! it ([`Model::recompile`]): only the universes that changed are built
//! again (seconds instead of a full compile for an edit to the reference
//! model).

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime};

use llevant::{Changed, Error, Model, Options, Progress, Result};

/// A model loaded staged ([`Store::load_staged`]): usable at once, its
/// octrees possibly still to build ([`Store::complete`]).
#[derive(Clone)]
pub struct Staged {
    pub model: Arc<Model>,
    /// What changed from the model it was recompiled from, if it was.
    pub changed: Option<Changed>,
    /// The entry to store the complete model as (none if read from the
    /// cache, or not caching).
    entry: Option<PathBuf>,
    path: PathBuf,
}

impl Staged {
    /// Whether the model is complete already (read from the cache, or
    /// nothing was left to build).
    pub fn is_complete(&self) -> bool {
        self.model.is_complete()
    }
}

/// Default budget for all entries, in GB (the reference model's entry takes
/// 1.6 GB).
const BUDGET_GB: f64 = 8.0;
const EXT: &str = "llevant";

/// Where compiled models are cached, if anywhere, and how much they may
/// take.
#[derive(Clone, Debug)]
pub struct Store {
    dir: Option<PathBuf>,
    budget_bytes: u64,
}

impl Store {
    /// The cache the environment asks for: none with `LLEVANT_CACHE=off`;
    /// else entries in `LLEVANT_CACHE_DIR`, or the user's cache directory
    /// (`%LOCALAPPDATA%\llevant\cache`, `$XDG_CACHE_HOME/llevant` or
    /// `~/.cache/llevant`), within `LLEVANT_CACHE_GB` (8 GB).
    pub fn from_env() -> Store {
        if std::env::var("LLEVANT_CACHE").as_deref() == Ok("off") {
            return Store::none();
        }
        let budget = std::env::var("LLEVANT_CACHE_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(BUDGET_GB);
        Store {
            dir: user_dir(),
            budget_bytes: (budget * 1e9) as u64,
        }
    }

    /// No cache: every load compiles.
    pub fn none() -> Store {
        Store {
            dir: None,
            budget_bytes: 0,
        }
    }

    /// Entries in `dir`, the most recently used kept within `budget_bytes`.
    pub fn at(dir: impl Into<PathBuf>, budget_bytes: u64) -> Store {
        Store {
            dir: Some(dir.into()),
            budget_bytes,
        }
    }

    /// The directory of the entries, if caching.
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Read and compile the MCNP input at `path` ([`Model::compile`]), or
    /// read the model compiled before from the cache, or recompile against
    /// the last version of `path` in the cache. With `progress`, each step
    /// is named in it (reading, the cache, then those of compiling,
    /// storing) and loading stops once it is cancelled.
    /// [`Model::timings`] say which happened: a cached model has a
    /// "saved model" step.
    pub fn load(&self, path: &Path, opts: &Options, progress: Option<&Progress>) -> Result<Model> {
        let staged = self.load_staged(path, opts, progress, None)?;
        let m = self.complete(staged, progress)?;
        Ok(Arc::into_inner(m).expect("the only reference to a model just loaded"))
    }

    /// [`Store::load`], returning as soon as the model can answer queries:
    /// a staged model whose octrees [`Store::complete`] builds (and then
    /// stores). Recompiles against `base` if given (a model of an earlier
    /// version of this input, in memory), else against the last entry of
    /// `path` in the cache, if any.
    pub fn load_staged(
        &self,
        path: &Path,
        opts: &Options,
        progress: Option<&Progress>,
        base: Option<&Model>,
    ) -> Result<Staged> {
        let none = Progress::new();
        let progress = progress.unwrap_or(&none);
        progress.set_stage("reading");
        let read_input = || {
            let t = Instant::now();
            let bytes = llevant::read_input_bytes(path)?;
            if progress.cancelled() {
                return Err(Error::Cancelled);
            }
            Ok((bytes, t.elapsed().as_secs_f64()))
        };
        let (mut bytes, mut read) = read_input()?;
        let mut entry = self.dir.as_ref().map(|d| d.join(key(&bytes, opts)));
        if let Some(e) = &entry {
            progress.set_stage("cache");
            // Not kept while the entry is decoded (378 MB for the reference
            // model): read again if the entry turns out to be missing.
            drop(bytes);
            if let Some(mut m) = read_entry(e) {
                m.timings_mut().insert(0, ("read", read));
                self.remember(path, e);
                return Ok(Staged {
                    changed: base.map(|b| Changed::between(b, &m)),
                    model: Arc::new(m),
                    entry: None,
                    path: path.to_path_buf(),
                });
            }
            progress.set_stage("reading");
            (bytes, read) = read_input()?;
            // The file may have changed since: keyed by what is compiled.
            entry = self.dir.as_ref().map(|d| d.join(key(&bytes, opts)));
        }
        let last = match base {
            Some(_) => None,
            None => self.last(path).and_then(|e| {
                progress.set_stage("reading the last version");
                let t = Instant::now();
                read_entry(&e).map(|m| (m, t.elapsed().as_secs_f64()))
            }),
        };
        let (mut m, changed) = match (base, &last) {
            (Some(b), _) | (None, Some((b, _))) => {
                let (m, c) = b.recompile(&bytes, opts, Some(progress))?;
                (m, Some(c))
            }
            (None, None) => (Model::compile_staged(&bytes, opts, Some(progress))?, None),
        };
        // The input is not needed any more: not kept while the octrees are
        // built and the model is written (378 MB for the reference model).
        drop(bytes);
        m.timings_mut().insert(0, ("read", read));
        if let Some((_, t)) = last {
            m.timings_mut().insert(1, ("last version", t));
        }
        Ok(Staged {
            model: Arc::new(m),
            changed,
            entry,
            path: path.to_path_buf(),
        })
    }

    /// The complete model of `staged`: its octrees built, then stored in
    /// the cache. Builds in place when `staged` holds the only reference to
    /// its model, else on a copy (sharing the octrees built already).
    pub fn complete(&self, staged: Staged, progress: Option<&Progress>) -> Result<Arc<Model>> {
        let Staged {
            model, entry, path, ..
        } = staged;
        if model.is_complete() && entry.is_none() {
            return Ok(model);
        }
        let mut m = match Arc::try_unwrap(model) {
            Ok(m) => m.into_complete(progress)?,
            Err(shared) => shared.complete(progress)?,
        };
        if let Some(entry) = &entry {
            if let Some(p) = progress {
                p.set_stage("storing in the cache");
            }
            let t = Instant::now();
            if self.store(entry, &m) {
                self.remember(&path, entry);
            }
            m.timings_mut()
                .push(("cache store", t.elapsed().as_secs_f64()));
        }
        Ok(Arc::new(m))
    }

    /// The file naming the last entry of input `path`.
    fn last_file(&self, path: &Path) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        let path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let name = blake3::hash(path.to_string_lossy().as_bytes()).to_hex();
        Some(dir.join("last").join(name.as_str()))
    }

    /// The last entry stored or read for input `path`, if it is still there.
    fn last(&self, path: &Path) -> Option<PathBuf> {
        let name = fs::read_to_string(self.last_file(path)?).ok()?;
        let entry = self.dir.as_ref()?.join(name.trim());
        entry.is_file().then_some(entry)
    }

    /// Remember `entry` as the last of input `path`.
    fn remember(&self, path: &Path, entry: &Path) {
        let (Some(file), Some(name)) = (self.last_file(path), entry.file_name()) else {
            return;
        };
        let Some(dir) = file.parent() else { return };
        if fs::create_dir_all(dir).is_err() {
            return;
        }
        let tmp = file.with_extension(format!("{}.tmp", std::process::id()));
        if fs::write(&tmp, name.to_string_lossy().as_bytes()).is_err()
            || fs::rename(&tmp, &file).is_err()
        {
            let _ = fs::remove_file(&tmp);
        }
    }

    /// Write `model` to `entry` (to a temporary file, then renamed, so a
    /// reader never sees half an entry), and drop the least recently used
    /// entries beyond the budget. Whether it was written.
    fn store(&self, entry: &Path, model: &Model) -> bool {
        let Some(dir) = entry.parent() else {
            return false;
        };
        if fs::create_dir_all(dir).is_err() {
            return false;
        }
        let tmp = entry.with_extension(format!("{}.tmp", std::process::id()));
        let written = File::create(&tmp).ok().and_then(|f| {
            let mut w = BufWriter::with_capacity(1 << 22, f);
            model.save(&mut w).ok()?;
            w.flush().ok()
        });
        if written.is_none() || fs::rename(&tmp, entry).is_err() {
            let _ = fs::remove_file(&tmp);
            return false;
        }
        let Ok(entries) = fs::read_dir(dir) else {
            return true;
        };
        let mut entries: Vec<(SystemTime, u64, PathBuf)> = entries
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == EXT))
            .filter_map(|e| {
                let m = e.metadata().ok()?;
                Some((m.modified().ok()?, m.len(), e.path()))
            })
            .collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.0));
        let mut used = 0u64;
        for (_, len, path) in entries {
            used += len;
            // The newest entry (just written) always stays.
            if used > self.budget_bytes && used != len {
                let _ = fs::remove_file(path);
            }
        }
        true
    }
}

/// The key of an input with these options, as a file name.
fn key(input: &[u8], opts: &Options) -> String {
    let mut h = blake3::Hasher::new();
    h.update(llevant::ENGINE_HASH.as_bytes());
    h.update(&llevant::FORMAT_VERSION.to_le_bytes());
    h.update(format!("{opts:?}").as_bytes());
    h.update_rayon(input);
    format!("{}.{EXT}", h.finalize().to_hex())
}

fn user_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("LLEVANT_CACHE_DIR") {
        return Some(PathBuf::from(d));
    }
    if cfg!(windows) {
        return std::env::var_os("LOCALAPPDATA")
            .map(|d| PathBuf::from(d).join("llevant").join("cache"));
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(d).join("llevant"));
    }
    std::env::var_os("HOME").map(|d| PathBuf::from(d).join(".cache").join("llevant"))
}

/// The model saved at `entry`, if there is a readable one. Decoded from a
/// memory map of the file: its bytes are the file's pages, not a copy of
/// them on the heap (1.6 GB for the reference model).
fn read_entry(entry: &Path) -> Option<Model> {
    // Writable, to mark it used (setting the time needs write access on
    // Windows); read-only if that is all there is.
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(entry)
        .or_else(|_| File::open(entry))
        .ok()?;
    let t = Instant::now();
    // SAFETY: entries are never changed in place: `Store::store` writes a
    // temporary file and renames it over the entry, which leaves this
    // mapping on the file it had. Another program writing into the cache
    // directory is outside what the cache supports.
    let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    let mut model = Model::from_bytes(&map).ok()?;
    drop(map);
    // As `Model::read` times it: the mapping and the decoding.
    model.timings_mut()[0].1 = t.elapsed().as_secs_f64();
    // Most recently used: kept longest.
    let _ = file.set_modified(SystemTime::now());
    Some(model)
}
