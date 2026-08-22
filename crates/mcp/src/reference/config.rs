use std::path::{Component, Path, PathBuf};

use super::{ReferenceError, ReferenceLayout};
use crate::config::EnvSource;

pub const REFERENCE_DATASET: &str = "fleet_reference";
const REFERENCE_ROOT_ENV: &str = "APEX_COGNEE_REFERENCE_ROOT";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceConfig {
    pub layout: ReferenceLayout,
    pub dataset: &'static str,
    pub limits: ReferenceLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceLimits {
    pub max_input_bytes: usize,
    pub max_batch_bytes: usize,
    pub max_batch_files: usize,
    pub max_pending_events: u64,
    pub max_pending_bytes: u64,
    pub max_item_bytes: usize,
    pub max_payload_bytes: usize,
}

impl Default for ReferenceLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 2 * 1024 * 1024,
            max_batch_bytes: 8 * 1024 * 1024,
            max_batch_files: 32,
            max_pending_events: 128,
            max_pending_bytes: 64 * 1024 * 1024,
            max_item_bytes: 2 * 1024,
            max_payload_bytes: 8 * 1024,
        }
    }
}

impl ReferenceConfig {
    pub fn from_env(env: &impl EnvSource) -> Result<Option<Self>, ReferenceError> {
        let Some(configured) = env
            .get(REFERENCE_ROOT_ENV)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let configured = PathBuf::from(configured);
        validate_configured_root(&configured)?;
        reject_symlinked_existing_ancestors(&configured)?;
        let root = if configured.exists() {
            configured
                .canonicalize()
                .map_err(|_| ReferenceError::InvalidRoot)?
        } else {
            configured
        };
        validate_configured_root(&root)?;
        Ok(Some(Self {
            layout: ReferenceLayout::under(root),
            dataset: REFERENCE_DATASET,
            limits: ReferenceLimits::default(),
        }))
    }
}

fn reject_symlinked_existing_ancestors(root: &Path) -> Result<(), ReferenceError> {
    let mut prefix = PathBuf::new();
    for component in root.components() {
        prefix.push(component.as_os_str());
        match std::fs::symlink_metadata(&prefix) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(ReferenceError::InvalidRoot);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(_) => return Err(ReferenceError::InvalidRoot),
        }
    }
    Ok(())
}

pub(crate) fn validate_configured_root(root: &Path) -> Result<(), ReferenceError> {
    if !root.is_absolute() || root.parent().is_none() {
        return Err(ReferenceError::InvalidRoot);
    }
    let mut normal_components = 0_usize;
    for component in root.components() {
        match component {
            Component::RootDir | Component::Prefix(_) => {}
            Component::Normal(_) => normal_components += 1,
            Component::CurDir | Component::ParentDir => return Err(ReferenceError::InvalidRoot),
        }
    }
    if normal_components == 0 {
        return Err(ReferenceError::InvalidRoot);
    }
    Ok(())
}
