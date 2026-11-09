use std::path::{Path, PathBuf};

use lofi_error::{Error, Result};

/// Stable per-workspace directory key shared by `sessions/` and `tmp/` so a
/// project's transcripts and scratch dirs land under the same label. Derived
/// from the workspace root path (not the basename alone, which would collide
/// across `~/work/a/b` and `~/other/a/b`) via a v5 (name-based, deterministic)
/// UUID — a v4 random id would change run-to-run and orphan prior state.
pub(crate) fn workspace_key(cwd: &Path) -> String {
    use std::os::unix::ffi::OsStrExt as _;

    let label = cwd
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            name.chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                        ch
                    } else {
                        '-'
                    }
                })
                .take(40)
                .collect::<String>()
        })
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "root".to_string());
    let id = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, cwd.as_os_str().as_bytes());
    format!("{label}-{id}")
}

/// Tests that mutate the process-global `XDG_STATE_HOME` / `LOFI_STATE_HOME`
/// env vars acquire this lock for their whole duration. The vars are
/// process-global, so parallel test threads otherwise interleave their
/// `set_var`/`remove_var` and race on the assertions. Shared with the
/// `models` tests that also point `XDG_STATE_HOME` at a temp dir.
#[cfg(test)]
pub(crate) static STATE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Precedence: `$LOFI_STATE_HOME` (used as the base, with `lofi` appended),
/// then `$XDG_STATE_HOME` on Linux via [`dirs::state_dir`], then a
/// `~/.local/state` fallback. The directory is *not* created here — call
/// [`ensure_state_dir`] before writing.
/// # Errors
/// Returns [`Error::State`] only when no base state directory can be
/// determined (e.g. `HOME` is unset and `dirs::state_dir` returns `None`).
pub fn state_dir() -> Result<PathBuf> {
    let mut base = std::env::var_os("LOFI_STATE_HOME").map(PathBuf::from);
    if base.is_none() {
        base = dirs::state_dir();
    }
    if base.is_none() {
        if let Ok(home) = std::env::var("HOME") {
            base = Some(PathBuf::from(home).join(".local").join("state"));
        }
    }
    let mut path = base.ok_or_else(|| Error::State("could not resolve state dir".into()))?;
    path.push("lofi");
    Ok(path)
}

/// # Errors
/// Propagates [`state_dir`]'s error if the base directory cannot be resolved.
pub fn discovery_cache_path() -> Result<PathBuf> {
    let mut path = state_dir()?;
    path.push("discovery.json");
    Ok(path)
}

/// # Errors
/// Returns [`Error::Io`] on filesystem failure, or [`Error::State`] if the
/// base directory cannot be resolved.
pub fn ensure_state_dir() -> Result<PathBuf> {
    let path = state_dir()?;
    ensure_private_dir(&path)?;
    Ok(path)
}

pub(crate) fn ensure_private_dir(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt as _, PermissionsExt as _};

    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true).mode(0o700).create(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
}

/// Tighten an existing directory in the state tree to mode 0700. A missing
/// directory stays missing: repair must not create one.
pub(crate) fn repair_private_dir(path: &std::path::Path) -> std::io::Result<()> {
    match std::fs::metadata(path) {
        Ok(_) => ensure_private_dir(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(crate) fn ensure_private_file(path: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

/// The workspace memory file `memory.md` in the workspace's directory
/// under `states/`, keyed like `sessions/` and `tmp/` so a project's
/// memory never crosses into another workspace. The agent maintains it
/// through the `lofi.memory` sandbox tool; the compaction hook reads it.
/// Nothing is created here: every session resolves this path at startup, so
/// creating the directory would leave one behind in a session that never
/// writes. The first write creates it, mode 0700 like the rest of the tree.
pub struct WorkspaceMemory {
    file: lofi_types::HostFile,
}

impl WorkspaceMemory {
    /// Resolve the memory file of the workspace rooted at `cwd`. Nothing is
    /// read or written yet.
    /// # Errors
    /// Returns [`Error::State`] if the base state directory cannot be
    /// resolved.
    pub fn open(cwd: &Path) -> Result<Self> {
        let mut path = state_dir()?;
        path.push("states");
        path.push(workspace_key(cwd));
        path.push("memory.md");
        Ok(Self {
            file: lofi_types::HostFile {
                path,
                dir_mode: 0o700,
                file_mode: 0o600,
            },
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.file.path
    }
}

impl From<WorkspaceMemory> for lofi_types::HostFile {
    fn from(memory: WorkspaceMemory) -> Self {
        memory.file
    }
}

/// The per-workspace state document: `state.json` in a directory under
/// `states/`, keyed like `sessions/` and `tmp/`, so a project's remembered
/// state never crosses into another workspace. The manager owns the document's format
/// and its read-modify-write cycle, so the rest of the code reaches state
/// through methods instead of reading the file itself.
pub struct WorkspaceState {
    path: PathBuf,
}

impl WorkspaceState {
    /// Open the state of the workspace rooted at `cwd`. Nothing is read or
    /// written yet: the file appears only when something is remembered.
    /// # Errors
    /// Returns [`Error::State`] if the base state directory cannot be
    /// resolved.
    pub fn open(cwd: &Path) -> Result<Self> {
        let mut path = state_dir()?;
        path.push("states");
        path.push(workspace_key(cwd));
        path.push("state.json");
        Ok(Self { path })
    }

    /// The remembered grants, longest-covering order preserved as stored.
    /// Reads fail toward "not approved": a missing, unreadable, or corrupt
    /// document only costs one more prompt.
    #[must_use]
    pub fn grants(&self) -> Vec<WorkspaceGrant> {
        std::fs::read(&self.path)
            .map(|bytes| parse_grants(&bytes))
            .unwrap_or_default()
    }

    /// The remembered grant that covers `dir` and allows `requested`.
    /// The longest match wins, so a nested grant names the narrowest root.
    #[must_use]
    pub fn grant_allowing(
        &self,
        dir: &std::path::Path,
        requested: lofi_types::Access,
    ) -> Option<WorkspaceGrant> {
        self.grants()
            .into_iter()
            .filter(|g| dir.starts_with(&g.dir) && g.access.allows(requested))
            .max_by_key(|g| g.dir.components().count())
    }

    /// The strongest remembered grant covering `dir`, whatever its level:
    /// what an upgrade prompt starts from.
    #[must_use]
    pub fn strongest_covering(&self, dir: &std::path::Path) -> Option<WorkspaceGrant> {
        self.grants()
            .into_iter()
            .filter(|g| dir.starts_with(&g.dir))
            .max_by_key(|g| (g.dir.components().count(), g.access))
    }

    /// Remember or update the grant for `dir`, so later sessions do not
    /// ask again.
    /// # Errors
    /// Returns [`Error::Io`] on filesystem failure, or [`Error::State`]
    /// if the document cannot be serialized.
    pub fn remember_grant(&self, dir: &std::path::Path, access: lofi_types::Access) -> Result<()> {
        self.update_grants(|grants| {
            if let Some(existing) = grants.iter_mut().find(|g| g.dir == dir) {
                existing.access = access;
            } else {
                grants.push(WorkspaceGrant {
                    dir: dir.to_path_buf(),
                    access,
                });
                grants.sort_by(|a, b| a.dir.cmp(&b.dir));
            }
        })
    }

    /// Forget the grant for exactly `dir`.
    /// # Errors
    /// Returns [`Error::Io`] on filesystem failure, or [`Error::State`]
    /// if the document cannot be serialized.
    pub fn forget_grant(&self, dir: &std::path::Path) -> Result<()> {
        self.update_grants(|grants| grants.retain(|g| g.dir != dir))
    }

    /// Read-modify-write under a lock: concurrent approvals touch the same
    /// file, so the existing list is re-read and merged inside the lock
    /// instead of trusting an earlier read.
    fn update_grants(&self, apply: impl FnOnce(&mut Vec<WorkspaceGrant>)) -> Result<()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A first grant has no file to merge into, but any other read failure
        // must abort the save: treating it as an empty list would overwrite
        // a good file and drop every remembered grant.
        let mut grants = match std::fs::read(&self.path) {
            Ok(bytes) => parse_grants(&bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.into()),
        };
        apply(&mut grants);
        self.write(&grants)
    }

    /// Replace the document atomically beside its destination, so a torn
    /// write cannot drop every remembered directory. The tmp name carries
    /// a timestamp so writers in other processes cannot clobber one tmp
    /// file before the rename.
    /// # Errors
    /// Returns [`Error::Io`] on filesystem failure, or [`Error::State`]
    /// if the state path has no parent.
    fn write(&self, grants: &[WorkspaceGrant]) -> Result<()> {
        let root = self
            .path
            .parent()
            .ok_or_else(|| Error::State("workspace state path has no parent".to_string()))?;
        ensure_private_dir(root)?;
        let value = serde_json::json!({ "version": 2, "grants": grants });
        let bytes = serde_json::to_vec(&value)
            .map_err(|e| Error::State(format!("serialize workspace state: {e}")))?;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let tmp = root.join(format!(".{}-{nanos}.tmp", std::process::id()));
        std::fs::write(&tmp, bytes)?;
        ensure_private_file(&tmp)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

/// One remembered grant as stored in the state document.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceGrant {
    #[serde(rename = "path")]
    pub dir: PathBuf,
    pub access: lofi_types::Access,
}

/// The parsed list, or empty on an unreadable or unknown-version document:
/// a bad file is treated as no grants rather than aborting the caller.
/// Version 1 carried bare directories with read-and-write meaning, so its
/// entries load as write grants; the file itself is only rewritten on the
/// next change.
fn parse_grants(bytes: &[u8]) -> Vec<WorkspaceGrant> {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return Vec::new();
    };
    match value.get("version").and_then(serde_json::Value::as_u64) {
        Some(1) => value
            .get("dirs")
            .and_then(serde_json::Value::as_array)
            .map(|dirs| {
                dirs.iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(|dir| WorkspaceGrant {
                        dir: PathBuf::from(dir),
                        access: lofi_types::Access::Write,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        Some(2) => value
            .get("grants")
            .and_then(serde_json::Value::as_array)
            .map(|grants| {
                grants
                    .iter()
                    .filter_map(|g| {
                        let dir = g.get("path")?.as_str()?.to_string();
                        let access = g
                            .get("access")
                            .and_then(serde_json::Value::as_str)
                            .and_then(|a| match a {
                                "read" => Some(lofi_types::Access::Read),
                                "write" => Some(lofi_types::Access::Write),
                                _ => None,
                            })?;
                        Some(WorkspaceGrant {
                            dir: PathBuf::from(dir),
                            access,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// The running image, held open before a rebuild can replace it. The kernel
/// reports a replaced binary as a path that is gone, which no child can
/// spawn; the open image outlives that.
///
/// The open has to happen early. On network and FUSE filesystems, such as a
/// virtiofs home, re-opening `/proc/self/exe` after the unlink fails with
/// `ESTALE`, while a descriptor taken beforehand keeps reading every byte.
#[cfg(target_os = "linux")]
static RUNNING_IMAGE: std::sync::Mutex<Option<PinnedImage>> = std::sync::Mutex::new(None);

#[cfg(target_os = "linux")]
struct PinnedImage {
    image: std::fs::File,
    /// The most recent copy, which later sessions link instead of writing the
    /// image again. It is only a source hint: once its session takes the copy
    /// away, the next session writes one.
    copy: Option<PathBuf>,
}

#[cfg(target_os = "linux")]
impl PinnedImage {
    /// A path to the pinned image inside `dir`, linked when that is free and
    /// written otherwise. A link shares the inode, so it outlives the session
    /// that wrote it and no later rebuild can change what it points at, while
    /// the install path stays a name a rebuild can take away or re-point.
    ///
    /// `None` only where `dir` cannot run a file at all: there the caller's
    /// own path is the one chance the child has, and it is kept while it still
    /// holds the running image.
    /// # Errors
    /// [`Error::Io`] when the image cannot be read, written, or marked
    /// executable, and [`Error::State`] when `dir` refuses to execute files
    /// and the caller's path no longer holds the image.
    fn path_in(&mut self, executable: &Path, dir: &Path) -> Result<Option<PathBuf>> {
        use std::io::{Seek as _, SeekFrom};

        // A file in `dir` is only worth preparing where it can be run. Where
        // it cannot, keeping the caller's path beats an error, and is all that
        // is left once a rebuild has taken it.
        if is_noexec(dir) {
            return without_exec(dir, shares_inode(&self.image, executable));
        }
        let path = dir.join("lofi");
        if let Some(copy) = &self.copy {
            if std::fs::hard_link(copy, &path).is_ok() {
                self.copy = Some(path.clone());
                return Ok(Some(path));
            }
        }
        // Linking the file at the install path costs nothing, and covers the
        // whole time before the first rebuild. The link is judged afterwards:
        // a rebuild can swap the file between the link and the check, and the
        // child must not end up running that new build against this process's
        // bridge.
        if std::fs::hard_link(executable, &path).is_ok() {
            if is_linked_image(&self.image, &path) {
                self.copy = Some(path.clone());
                return Ok(Some(path));
            }
            // The link caught another build, so it was this call that made the
            // path: the copy below takes it back.
            std::fs::remove_file(&path)?;
        }
        // The handle is shared, so every copy starts from its beginning.
        self.image.seek(SeekFrom::Start(0))?;
        let copy = copy_image(&mut self.image, dir)?;
        self.copy = Some(copy.clone());
        Ok(Some(copy))
    }
}

/// The path the child gets when `dir` cannot execute files: the caller's own
/// while that still holds the image, and an error once a rebuild took it.
#[cfg(target_os = "linux")]
fn without_exec(dir: &Path, caller_holds_image: bool) -> Result<Option<PathBuf>> {
    if caller_holds_image {
        return Ok(None);
    }
    Err(Error::State(format!(
        "the session directory {} is on a noexec mount, so the running image \
         cannot be executed from it",
        dir.display()
    )))
}

/// Whether `dir` sits on a mount that refuses to execute files. A mount that
/// cannot be asked is treated as executable, so the copy speaks for itself.
#[cfg(target_os = "linux")]
fn is_noexec(dir: &Path) -> bool {
    use nix::sys::statvfs::{statvfs, FsFlags};

    match statvfs(dir) {
        Ok(stat) => stat.flags().contains(FsFlags::ST_NOEXEC),
        Err(_) => false,
    }
}

/// Whether the file just linked at `path` is the image `handle` holds open. A
/// symlink is refused: it names no inode of its own, so a rebuild could
/// re-point it.
#[cfg(target_os = "linux")]
fn is_linked_image(handle: &std::fs::File, path: &Path) -> bool {
    let (Ok(image), Ok(file)) = (handle.metadata(), std::fs::symlink_metadata(path)) else {
        return false;
    };
    !file.file_type().is_symlink() && same_inode(&image, &file)
}

/// Whether `path` resolves to the image `handle` holds open.
#[cfg(target_os = "linux")]
fn shares_inode(handle: &std::fs::File, path: &Path) -> bool {
    let (Ok(image), Ok(file)) = (handle.metadata(), std::fs::metadata(path)) else {
        return false;
    };
    same_inode(&image, &file)
}

/// Whether two handles name one file.
#[cfg(target_os = "linux")]
fn same_inode(image: &std::fs::Metadata, file: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt as _;

    (image.dev(), image.ino()) == (file.dev(), file.ino())
}

/// The pinned image slot, shared by every session in this process.
#[cfg(target_os = "linux")]
fn running_image() -> std::sync::MutexGuard<'static, Option<PinnedImage>> {
    RUNNING_IMAGE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Hold the running image open, so a later rebuild cannot take its
/// `/proc/self/exe` away. Elsewhere the open is not possible and the caller
/// keeps using its own path, which a rebuild replaces rather than removes.
pub fn pin_running_image() {
    #[cfg(target_os = "linux")]
    {
        if let Ok(image) = std::fs::File::open("/proc/self/exe") {
            let mut slot = running_image();
            if slot.is_none() {
                *slot = Some(PinnedImage { image, copy: None });
            }
        }
    }
}

/// A path to the running image inside `dir`, for a child that spawns it: a
/// rebuild can take the caller's own path away, and can swap the build behind
/// it before the child execs.
///
/// `None` when nothing was pinned, and when `dir` cannot execute files while
/// the caller's own path still holds the pinned image: the caller then keeps
/// its own path.
/// # Errors
/// [`Error::Io`] when the image cannot be read, written, or marked
/// executable, and [`Error::State`] when `dir` refuses to execute files.
#[cfg_attr(not(target_os = "linux"), allow(clippy::unnecessary_wraps))]
pub(crate) fn running_image_path(executable: &Path, dir: &Path) -> Result<Option<PathBuf>> {
    #[cfg(target_os = "linux")]
    {
        let mut slot = running_image();
        let Some(pinned) = slot.as_mut() else {
            return Ok(None);
        };
        pinned.path_in(executable, dir)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (executable, dir);
        Ok(None)
    }
}

/// Write an open image into `dir` as a private executable file. A held fd
/// reads on after its path is gone, which is what a rebuild leaves behind.
#[cfg(target_os = "linux")]
fn copy_image(image: &mut std::fs::File, dir: &Path) -> Result<PathBuf> {
    use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

    let path = dir.join("lofi");
    // The session dir is fresh, so the copy claims the path rather than
    // replacing one: a file already there is a bug worth an error, and
    // truncating a link to the shared image would empty it for every session.
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o700)
        .open(&path)?;
    std::io::copy(image, &mut out)?;
    // Closing is enough: the page cache already holds the bytes for the child
    // that execs the copy, and fsyncing tens of megabytes would land on every
    // session open.
    drop(out);
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700))?;
    Ok(path)
}

pub(crate) struct SessionTempDir {
    path: PathBuf,
    /// The workspace-key directory that groups this workspace's leases. It
    /// holds nothing of its own, so it goes with the last lease.
    group: Option<PathBuf>,
    /// The lease file, opened on the first call to [`Self::ensure`]. Until
    /// then nothing exists on disk: a session that writes no pageable output
    /// and starts no job leaves no directory at all.
    lease: std::sync::Mutex<Option<std::fs::File>>,
}

impl SessionTempDir {
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Create the directory and take its lease, once. Every writer into the
    /// directory calls this first.
    pub(crate) fn ensure(&self) -> std::io::Result<()> {
        use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};

        let mut lease = self
            .lease
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if lease.is_some() {
            return Ok(());
        }
        if let Some(group) = &self.group {
            if let Some(root) = group.parent() {
                ensure_private_dir(root)?;
            }
            ensure_private_dir(group)?;
        }
        // Stage the directory with its lease inside, then rename it into
        // place: a sweeper in another process must never see it without a
        // lease and collect it.
        let staging = self
            .path
            .with_file_name(format!(".creating-{}", uuid::Uuid::new_v4().simple()));
        std::fs::DirBuilder::new().mode(0o700).create(&staging)?;
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(staging.join(".lease"))
        {
            Ok(file) => file,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&staging);
                return Err(error);
            }
        };
        if let Err(error) = file.lock() {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error);
        }
        if let Err(error) = std::fs::rename(&staging, &self.path) {
            let _ = std::fs::remove_dir_all(&staging);
            return Err(error);
        }
        *lease = Some(file);
        Ok(())
    }

    #[cfg(test)]
    fn create_under(root: &std::path::Path) -> Self {
        Self {
            path: root.join(uuid::Uuid::new_v4().simple().to_string()),
            group: None,
            lease: std::sync::Mutex::new(None),
        }
    }
}

impl Drop for SessionTempDir {
    fn drop(&mut self) {
        let lease = self
            .lease
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if lease.is_none() {
            return;
        }
        let _ = std::fs::remove_dir_all(&self.path);
        // The scratch tree carries nothing of its own, so it falls away with
        // the last lease. Each removal fails while a sibling still holds the
        // directory, which is exactly when it must stay.
        if let Some(group) = &self.group {
            let _ = std::fs::remove_dir(group);
            if let Some(root) = group.parent() {
                let _ = std::fs::remove_dir(root);
            }
        }
    }
}

const LEGACY_TMP_MAX_AGE: std::time::Duration = std::time::Duration::from_hours(24);

fn entry_is_stale(entry: &std::fs::DirEntry) -> bool {
    entry
        .metadata()
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|age| age >= LEGACY_TMP_MAX_AGE)
}

/// `tmp/` holds one workspace-key dir per project, each containing leased
/// session dirs; leased dirs never sit directly at the root, so this recurses
/// exactly one level. A workspace-key dir has no `.lease` of its own — it is
/// swept into and dropped once empty.
fn collect_abandoned_tmp_dirs(root: &std::path::Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(".creating-")
        {
            if entry_is_stale(&entry) {
                let _ = std::fs::remove_dir_all(path);
            }
            continue;
        }
        // Workspace-key dir: holds leased session dirs, not a `.lease` of its
        // own; sweep into it and drop it once empty.
        if !path.join(".lease").exists() {
            let _ = collect_abandoned_tmp_dirs(&path);
            let _ = std::fs::remove_dir(&path);
            continue;
        }
        let lease_path = path.join(".lease");
        match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(lease_path)
        {
            Ok(lease) if lease.try_lock().is_ok() => {
                let _ = std::fs::remove_dir_all(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if entry_is_stale(&entry) {
                    let _ = std::fs::remove_dir_all(path);
                }
            }
            Ok(_) | Err(_) => {}
        }
    }
    Ok(())
}

/// The per-session directory for pageable tool output and job logs, scoped to
/// the workspace so a project's scratch dirs group alongside its transcripts.
/// Nothing is created here: the directory and its lease arrive with the first
/// write, so a session that writes none leaves no directory behind. Resolving
/// it tidies the existing tree instead.
/// # Errors
/// Returns [`Error::State`] if the base directory cannot be resolved.
pub(crate) fn create_session_tmp_dir(cwd: &Path) -> Result<SessionTempDir> {
    let home = state_dir()?;
    tidy_state_tree(&home);
    let group = home.join("tmp").join(workspace_key(cwd));
    Ok(SessionTempDir {
        path: group.join(uuid::Uuid::new_v4().simple().to_string()),
        group: Some(group),
        lease: std::sync::Mutex::new(None),
    })
}

/// Tighten the modes of an existing state tree and collect the scratch dirs
/// a dead session left behind. Nothing is created: a session that writes
/// nothing must leave no directory, and every session resolves this state at
/// startup.
fn tidy_state_tree(home: &Path) {
    let scratch = home.join("tmp");
    for dir in [
        home.join("states"),
        home.join("sessions"),
        home.join("logs"),
        scratch.clone(),
    ] {
        let _ = repair_private_dir(&dir);
    }
    let _ = collect_abandoned_tmp_dirs(&scratch);
    let _ = repair_private_dir(home);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// A test temp dir runs files on most hosts, but `/tmp` is noexec on some,
    /// and the ladder prepares nothing there.
    #[cfg(target_os = "linux")]
    fn skip_without_exec(dir: &Path) -> bool {
        if is_noexec(dir) {
            eprintln!("skipping: {} is on a noexec mount", dir.display());
            return true;
        }
        false
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn paths_under_xdg_state_home() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join("lofi_state_test_paths");
        std::env::set_var("XDG_STATE_HOME", &tmp);

        let sd = state_dir().unwrap();
        assert!(sd.ends_with("lofi"));
        assert!(sd.starts_with(&tmp));

        let dc = discovery_cache_path().unwrap();
        assert!(dc.ends_with("lofi/discovery.json"));
        assert!(dc.starts_with(&tmp));

        std::env::remove_var("XDG_STATE_HOME");
    }

    #[test]
    fn lofi_state_home_overrides_xdg() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join("lofi_state_test_lofi_override");
        std::env::set_var("XDG_STATE_HOME", "/this/should/not/be/used");
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        let sd = state_dir().unwrap();
        assert_eq!(sd, tmp.join("lofi"));

        let dc = discovery_cache_path().unwrap();
        assert_eq!(dc, tmp.join("lofi").join("discovery.json"));

        std::env::remove_var("LOFI_STATE_HOME");
        std::env::remove_var("XDG_STATE_HOME");
    }

    #[test]
    fn workspace_memory_is_workspace_scoped() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp =
            std::env::temp_dir().join(format!("lofi_state_test_memory_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        let memory = WorkspaceMemory::open(Path::new("/ws/proj")).unwrap();
        let path = memory.path();
        assert!(path.ends_with("memory.md"));
        assert!(path.starts_with(&tmp));
        let key = path
            .parent()
            .and_then(std::path::Path::file_name)
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert_eq!(key, workspace_key(Path::new("/ws/proj")));
        // Another workspace gets its own memory file.
        let other = WorkspaceMemory::open(Path::new("/ws/other")).unwrap();
        assert_ne!(other.path(), path);

        std::env::remove_var("LOFI_STATE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn workspace_memory_open_creates_nothing() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "lofi_state_test_memory_lazy_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        let memory = WorkspaceMemory::open(Path::new("/ws/proj")).unwrap();
        let path = memory.path();
        assert!(!path.exists());
        assert!(!path.parent().unwrap().exists());
        assert!(
            !super::state_dir().unwrap().exists(),
            "resolving the memory path must not create the state directory"
        );

        std::env::remove_var("LOFI_STATE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A pinned image over a small file, so the tests stay cheap and own all
    /// of their state instead of the process-global slot.
    #[cfg(target_os = "linux")]
    fn pinned(path: &Path) -> PinnedImage {
        PinnedImage {
            image: std::fs::File::open(path).unwrap(),
            copy: None,
        }
    }

    #[cfg(target_os = "linux")]
    fn inode(path: &Path) -> u64 {
        use std::os::unix::fs::MetadataExt as _;
        std::fs::metadata(path).unwrap().ino()
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_noexec_dir_keeps_the_callers_path_only_while_it_holds_the_image() {
        // No mount is needed: the branch is asked directly, because a test
        // cannot mount a noexec filesystem.
        let dir = tempfile::tempdir().unwrap();
        assert!(without_exec(dir.path(), true).unwrap().is_none());
        let error = without_exec(dir.path(), false).unwrap_err().to_string();
        assert!(error.contains("noexec"), "{error}");
        assert!(error.contains(&dir.path().display().to_string()), "{error}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn an_untouched_install_path_is_linked_without_a_copy() {
        let dir = tempfile::tempdir().unwrap();
        let installed = dir.path().join("installed");
        std::fs::write(&installed, b"image").unwrap();
        let mut pinned = pinned(&installed);

        let session = tempfile::tempdir().unwrap();
        if skip_without_exec(session.path()) {
            return;
        }
        let path = pinned.path_in(&installed, session.path()).unwrap().unwrap();
        // The link shares the inode, so no copy was written for the session.
        assert_eq!(inode(&path), inode(&installed));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_rebuild_at_the_install_path_is_not_what_the_child_runs() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("image");
        std::fs::write(&image, b"running build").unwrap();
        let mut pinned = pinned(&image);
        // What a rebuild leaves at the install path: another build, another
        // inode.
        let installed = dir.path().join("lofi");
        std::fs::write(&installed, b"new build!!").unwrap();

        let session = tempfile::tempdir().unwrap();
        if skip_without_exec(session.path()) {
            return;
        }
        let path = pinned.path_in(&installed, session.path()).unwrap().unwrap();
        assert_ne!(inode(&path), inode(&installed));
        assert_eq!(std::fs::read(&path).unwrap(), b"running build");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_copy_is_written_again_once_its_lease_is_gone() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("image");
        std::fs::write(&image, b"image").unwrap();
        let mut pinned = pinned(&image);
        // The install path is gone, as it is after a rebuild.
        let gone = dir.path().join("lofi (deleted)");

        let first = tempfile::tempdir().unwrap();
        if skip_without_exec(first.path()) {
            return;
        }
        let first_copy = pinned.path_in(&gone, first.path()).unwrap().unwrap();
        assert_eq!(std::fs::read(&first_copy).unwrap(), b"image");
        // The next session links that copy rather than writing its own.
        let second = tempfile::tempdir().unwrap();
        let second_copy = pinned.path_in(&gone, second.path()).unwrap().unwrap();
        assert_eq!(inode(&first_copy), inode(&second_copy));

        // The session holding the most recent copy takes it away, so the next
        // one has to read the pinned handle again, from its beginning.
        let linked = inode(&second_copy);
        drop(second);
        let third = tempfile::tempdir().unwrap();
        let third_copy = pinned.path_in(&gone, third.path()).unwrap().unwrap();
        assert_ne!(inode(&third_copy), linked);
        assert_eq!(std::fs::read(&third_copy).unwrap(), b"image");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_held_image_copies_after_its_path_is_removed() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("image");
        std::fs::write(&source, b"#!/bin/sh\nprintf image\n").unwrap();
        let mut image = std::fs::File::open(&source).unwrap();
        std::fs::remove_file(&source).unwrap();

        let path = copy_image(&mut image, dir.path()).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"#!/bin/sh\nprintf image\n");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    fn scratch_lease(group: &std::path::Path) -> SessionTempDir {
        SessionTempDir {
            path: group.join(uuid::Uuid::new_v4().simple().to_string()),
            group: Some(group.to_path_buf()),
            lease: std::sync::Mutex::new(None),
        }
    }

    #[test]
    fn the_scratch_lease_creates_nothing_until_it_is_used() {
        let root = tempfile::tempdir().unwrap();
        let group = root.path().join("lofi").join("tmp").join("proj-key");
        let lease = scratch_lease(&group);

        assert!(!lease.path().exists());
        assert!(!group.exists());
        drop(lease);
        assert!(!root.path().join("lofi").exists());
    }

    #[test]
    fn the_last_lease_takes_the_scratch_tree_with_it() {
        let root = tempfile::tempdir().unwrap();
        let scratch = root.path().join("lofi").join("tmp");
        let group = scratch.join("proj-key");

        let lease = scratch_lease(&group);
        lease.ensure().unwrap();
        assert!(lease.path().join(".lease").exists());
        drop(lease);

        assert!(!group.exists());
        assert!(!scratch.exists());
        assert!(root.path().exists());
    }

    #[test]
    fn a_sibling_lease_keeps_the_scratch_tree() {
        let root = tempfile::tempdir().unwrap();
        let scratch = root.path().join("lofi").join("tmp");
        let group = scratch.join("proj-key");

        let first = scratch_lease(&group);
        let second = scratch_lease(&group);
        first.ensure().unwrap();
        second.ensure().unwrap();
        drop(first);
        assert!(group.exists());
        drop(second);
        assert!(!group.exists());
        assert!(!scratch.exists());
    }

    #[test]
    fn workspace_key_is_readable_and_collision_resistant() {
        let first = workspace_key(Path::new("/work/a-b/c"));
        let second = workspace_key(Path::new("/work/a/b-c"));
        assert!(first.starts_with("c-"));
        assert!(second.starts_with("b-c-"));
        assert_ne!(first, second);
        assert_eq!(workspace_key(Path::new("/work/a-b/c")), first);
    }

    #[test]
    fn session_tmp_dir_is_removed_after_last_lease() {
        let root = tempfile::tempdir().unwrap();
        let lease = SessionTempDir::create_under(root.path());
        lease.ensure().unwrap();
        let path = lease.path().to_path_buf();
        std::fs::write(path.join("output.log"), "secret").unwrap();
        let lease = std::sync::Arc::new(lease);
        let clone = lease.clone();

        drop(lease);
        assert!(path.exists());
        drop(clone);
        assert!(!path.exists());
    }

    #[test]
    fn startup_collects_unlocked_temp_dirs() {
        use std::os::unix::fs::OpenOptionsExt as _;

        let root = tempfile::tempdir().unwrap();
        let abandoned = root.path().join("abandoned");
        ensure_private_dir(&abandoned).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(abandoned.join(".lease"))
            .unwrap();

        collect_abandoned_tmp_dirs(root.path()).unwrap();
        assert!(!abandoned.exists());
    }

    #[test]
    fn startup_preserves_locked_temp_dirs() {
        let root = tempfile::tempdir().unwrap();
        let lease = SessionTempDir::create_under(root.path());
        lease.ensure().unwrap();
        let path = lease.path().to_path_buf();

        collect_abandoned_tmp_dirs(root.path()).unwrap();
        assert!(path.exists());

        drop(lease);
        assert!(!path.exists());
    }

    #[test]
    fn collect_sweeps_workspace_scoped_abandoned_dirs() {
        let root = tempfile::tempdir().unwrap();
        let scoped = root.path().join(workspace_key(Path::new("/work/proj")));
        let abandoned = scoped.join("abandoned");
        ensure_private_dir(&abandoned).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(abandoned.join(".lease"))
            .unwrap();

        collect_abandoned_tmp_dirs(root.path()).unwrap();
        assert!(!scoped.exists());
    }

    #[test]
    fn startup_preserves_fresh_staging_dirs() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join(".creating-in-progress");
        ensure_private_dir(&staging).unwrap();

        collect_abandoned_tmp_dirs(root.path()).unwrap();
        assert!(staging.exists());
    }

    #[test]
    fn grants_remember_across_restarts_and_stay_workspace_scoped() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp =
            std::env::temp_dir().join(format!("lofi_state_test_approved_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        let ws = Path::new("/ws/proj");
        let state = WorkspaceState::open(ws).unwrap();
        let read = lofi_types::Access::Read;
        let write = lofi_types::Access::Write;
        assert_eq!(state.grant_allowing(Path::new("/data/dir"), read), None);

        state.remember_grant(Path::new("/data/dir"), write).unwrap();

        // Every lookup re-reads the file, so these cover a restart.
        assert_eq!(
            state.grant_allowing(Path::new("/data/dir"), write),
            Some(WorkspaceGrant {
                dir: PathBuf::from("/data/dir"),
                access: write
            })
        );
        assert_eq!(
            state.grant_allowing(Path::new("/data/dir/sub/x.txt"), read),
            Some(WorkspaceGrant {
                dir: PathBuf::from("/data/dir"),
                access: write
            })
        );
        // A write grant allows reads, but a read grant never allows writes.
        assert_eq!(
            state
                .grant_allowing(Path::new("/data/dir"), write)
                .unwrap()
                .access,
            write
        );
        state.remember_grant(Path::new("/ro"), read).unwrap();
        assert_eq!(state.grant_allowing(Path::new("/ro/sub"), write), None);
        assert_eq!(
            state
                .grant_allowing(Path::new("/ro/sub"), read)
                .unwrap()
                .access,
            read
        );

        // A nested grant wins over its parent, so the narrowest root names
        // the tree the tool result is shown against.
        state.remember_grant(Path::new("/data"), read).unwrap();
        assert_eq!(
            state.grant_allowing(Path::new("/data/dir/sub"), read),
            Some(WorkspaceGrant {
                dir: PathBuf::from("/data/dir"),
                access: write
            })
        );
        assert_eq!(
            state.grant_allowing(Path::new("/data/other"), read),
            Some(WorkspaceGrant {
                dir: PathBuf::from("/data"),
                access: read
            })
        );

        // A sibling of every grant stays unapproved, and another workspace
        // remembers nothing.
        assert_eq!(state.grant_allowing(Path::new("/elsewhere"), read), None);
        assert_eq!(
            WorkspaceState::open(Path::new("/ws/other"))
                .unwrap()
                .grant_allowing(Path::new("/data/dir"), read),
            None
        );

        // Forgetting removes exactly one grant; the nested one stays.
        state.forget_grant(Path::new("/data")).unwrap();
        assert_eq!(
            state
                .strongest_covering(Path::new("/data/dir/sub"))
                .unwrap()
                .dir,
            PathBuf::from("/data/dir")
        );

        std::env::remove_var("LOFI_STATE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn version_one_dirs_load_as_write_grants() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "lofi_state_test_approved_v1_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        let ws = Path::new("/ws/proj");
        let state = WorkspaceState::open(ws).unwrap();
        let path = state.path.clone();
        ensure_private_dir(path.parent().unwrap()).unwrap();
        std::fs::write(&path, br#"{"version":1,"dirs":["/data/dir"]}"#).unwrap();

        // The old format granted read and write together, so it loads as
        // write access without rewriting the file.
        assert_eq!(
            state.grant_allowing(Path::new("/data/dir"), lofi_types::Access::Write),
            Some(WorkspaceGrant {
                dir: PathBuf::from("/data/dir"),
                access: lofi_types::Access::Write
            })
        );
        assert!(std::fs::read_to_string(&path)
            .unwrap()
            .contains(r#""version":1"#));

        std::env::remove_var("LOFI_STATE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn approved_roots_survive_a_corrupt_file() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "lofi_state_test_approved_corrupt_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        let ws = Path::new("/ws/proj");
        let state = WorkspaceState::open(ws).unwrap();
        let path = state.path.clone();
        ensure_private_dir(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{ not json").unwrap();

        assert_eq!(
            state.grant_allowing(Path::new("/data/dir"), lofi_types::Access::Read),
            None
        );
        state
            .remember_grant(Path::new("/data/dir"), lofi_types::Access::Read)
            .unwrap();
        assert_eq!(
            state.grant_allowing(Path::new("/data/dir"), lofi_types::Access::Read),
            Some(WorkspaceGrant {
                dir: PathBuf::from("/data/dir"),
                access: lofi_types::Access::Read
            })
        );

        std::env::remove_var("LOFI_STATE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn remember_aborts_when_the_existing_list_cannot_be_read() {
        let _env = super::STATE_ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "lofi_state_test_approved_unreadable_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::env::set_var("LOFI_STATE_HOME", &tmp);

        // A directory at the file path makes the merge read fail with an
        // error other than NotFound.
        let ws = Path::new("/ws/proj");
        let state = WorkspaceState::open(ws).unwrap();
        let path = state.path.clone();
        ensure_private_dir(&path).unwrap();

        // The save must abort rather than merge into an empty list and
        // overwrite the file.
        assert!(state
            .remember_grant(Path::new("/data/dir"), lofi_types::Access::Read)
            .is_err());
        assert!(path.is_dir());

        std::env::remove_var("LOFI_STATE_HOME");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
