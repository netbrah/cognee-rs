//! Private per-engineer state paths and file creation helpers.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateLayout {
    pub root: PathBuf,
    pub system: PathBuf,
    pub data: PathBuf,
    pub cache: PathBuf,
    pub spool_pending: PathBuf,
    pub spool_processing: PathBuf,
    pub spool_failed: PathBuf,
    pub locks: PathBuf,
    pub context: PathBuf,
    pub status: PathBuf,
    pub ledger: PathBuf,
}

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("private state I/O failed")]
    Io(#[source] std::io::Error),
    #[error("private file must remain below the state root")]
    OutsideRoot,
}

impl From<std::io::Error> for LayoutError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl StateLayout {
    pub fn under(root: PathBuf) -> Self {
        Self {
            system: root.join("system"),
            data: root.join("data"),
            cache: root.join("cache"),
            spool_pending: root.join("spool/pending"),
            spool_processing: root.join("spool/processing"),
            spool_failed: root.join("spool/failed"),
            locks: root.join("locks"),
            context: root.join("context"),
            status: root.join("status"),
            ledger: root.join("ledger"),
            root,
        }
    }

    pub fn ensure_private(&self) -> Result<(), LayoutError> {
        let spool = self.root.join("spool");
        let logs = self.status.join("logs");
        for directory in [
            &self.root,
            &self.system,
            &self.data,
            &self.cache,
            &spool,
            &self.spool_pending,
            &self.spool_processing,
            &self.spool_failed,
            &self.locks,
            &self.context,
            &self.status,
            &logs,
            &self.ledger,
        ] {
            fs::create_dir_all(directory)?;
            set_directory_private(directory)?;
        }
        Ok(())
    }

    pub fn write_private_file(&self, path: &Path, contents: &[u8]) -> Result<(), LayoutError> {
        let relative = path
            .strip_prefix(&self.root)
            .map_err(|_| LayoutError::OutsideRoot)?;
        if relative.as_os_str().is_empty()
            || !relative
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_)))
        {
            return Err(LayoutError::OutsideRoot);
        }
        let parent = path.parent().ok_or(LayoutError::OutsideRoot)?;
        fs::create_dir_all(parent)?;
        make_ancestors_private(&self.root, parent)?;

        let mut options = OpenOptions::new();
        options.create(true).truncate(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        file.write_all(contents)?;
        file.sync_all()?;
        set_file_private(path)?;
        Ok(())
    }
}

fn make_ancestors_private(root: &Path, leaf: &Path) -> Result<(), LayoutError> {
    let relative = leaf
        .strip_prefix(root)
        .map_err(|_| LayoutError::OutsideRoot)?;
    let mut current = root.to_path_buf();
    set_directory_private(&current)?;
    for component in relative.components() {
        current.push(component);
        set_directory_private(&current)?;
    }
    Ok(())
}

#[cfg(unix)]
fn set_directory_private(path: &Path) -> Result<(), LayoutError> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_directory_private(_path: &Path) -> Result<(), LayoutError> {
    Ok(())
}

#[cfg(unix)]
fn set_file_private(path: &Path) -> Result<(), LayoutError> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_file_private(_path: &Path) -> Result<(), LayoutError> {
    Ok(())
}
