//! Durable, bounded event spool shared by hooks and transient workers.

use std::fs::{self};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::DateTime;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::atomic_fs::{
    AtomicFsError, AtomicWriteOutcome, ReplaceMode, SyncOps, SystemSyncOps,
    ensure_private_directory, remove_durable, rename_durable, write_atomic,
};
use crate::event::EventEnvelope;
use crate::layout::{LayoutError, StateLayout};
use crate::limits::ResourceLimits;

pub const MAX_EVENT_FILE_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    High,
    Normal,
    Low,
}

impl Priority {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "high" => Some(Self::High),
            "normal" => Some(Self::Normal),
            "low" => Some(Self::Low),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpoolRecord {
    #[serde(flatten)]
    pub envelope: EventEnvelope,
    #[serde(default)]
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub not_before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error_class: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpoolFile {
    pub path: PathBuf,
    pub priority: Priority,
    pub source_unix_nanos: i64,
    pub event_id: String,
}

#[derive(Debug)]
pub struct ClaimedEvent {
    path: PathBuf,
    file_name: String,
    pub record: SpoolRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnqueueOutcome {
    pub path: PathBuf,
    pub duplicate: bool,
    pub capture_degraded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureDisposition {
    Requeued(u32),
    Quarantined(u32),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpoolDepths {
    pub pending: usize,
    pub processing: usize,
    pub failed: usize,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub requeued: usize,
    pub committed_removed: usize,
    pub invalid_quarantined: usize,
}

#[derive(Debug, Error)]
pub enum SpoolError {
    #[error("spool I/O failed")]
    Io(#[source] io::Error),
    #[error("spool atomic operation failed")]
    Atomic(#[source] AtomicFsError),
    #[error("private state layout failed")]
    Layout(#[source] LayoutError),
    #[error("spool event JSON is invalid")]
    Json(#[source] serde_json::Error),
    #[error("spool event timestamp is invalid")]
    InvalidTimestamp,
    #[error("spool event path is invalid")]
    InvalidPath,
    #[error("spool event is larger than the configured cap")]
    EventTooLarge,
    #[error("spool event identity does not match its durable file")]
    IdentityMismatch,
    #[error("spool retry count overflowed")]
    AttemptOverflow,
}

impl From<io::Error> for SpoolError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<AtomicFsError> for SpoolError {
    fn from(error: AtomicFsError) -> Self {
        Self::Atomic(error)
    }
}

impl From<LayoutError> for SpoolError {
    fn from(error: LayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<serde_json::Error> for SpoolError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone)]
pub struct Spool {
    layout: StateLayout,
    limits: ResourceLimits,
    sync: Arc<dyn SyncOps>,
}

impl Spool {
    pub fn new(layout: StateLayout, limits: ResourceLimits) -> Self {
        Self::with_sync(layout, limits, Arc::new(SystemSyncOps))
    }

    pub fn with_sync(layout: StateLayout, limits: ResourceLimits, sync: Arc<dyn SyncOps>) -> Self {
        Self {
            layout,
            limits,
            sync,
        }
    }

    pub fn enqueue(
        &self,
        envelope: &EventEnvelope,
        priority: Priority,
    ) -> Result<EnqueueOutcome, SpoolError> {
        self.layout.ensure_private()?;
        let source_unix_nanos = source_unix_nanos(&envelope.timestamp)?;
        let file_name = event_file_name(priority, source_unix_nanos, &envelope.event_id);
        let path = self.layout.spool_pending.join(file_name);
        if path.exists() {
            return Ok(EnqueueOutcome {
                path,
                duplicate: true,
                capture_degraded: false,
            });
        }

        let capture_degraded = self.spool_bytes()? >= self.limits.spool_high_water_bytes;
        let durable_envelope = if capture_degraded {
            degraded_envelope(envelope)
        } else {
            envelope.clone()
        };
        let record = SpoolRecord {
            envelope: durable_envelope,
            attempts: 0,
            not_before: None,
            last_error_class: None,
        };
        let contents = serde_json::to_vec(&record)?;
        if contents.len() as u64 > MAX_EVENT_FILE_BYTES {
            return Err(SpoolError::EventTooLarge);
        }
        let outcome = write_atomic(&path, &contents, ReplaceMode::NoReplace, self.sync.as_ref())?;
        Ok(EnqueueOutcome {
            path,
            duplicate: outcome == AtomicWriteOutcome::Existing,
            capture_degraded,
        })
    }

    pub fn pending_files(&self) -> Result<Vec<SpoolFile>, SpoolError> {
        self.layout.ensure_private()?;
        let mut files = Vec::new();
        for entry in fs::read_dir(&self.layout.spool_pending)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                continue;
            }
            if let Some(parsed) = parse_event_file(&entry.path()) {
                files.push(parsed);
            }
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(files)
    }

    pub fn claim(&self, pending: &SpoolFile) -> Result<ClaimedEvent, SpoolError> {
        if pending.path.parent() != Some(self.layout.spool_pending.as_path()) {
            return Err(SpoolError::InvalidPath);
        }
        let file_name = pending
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(SpoolError::InvalidPath)?
            .to_owned();
        let processing_path = self.layout.spool_processing.join(&file_name);
        rename_durable(&pending.path, &processing_path, self.sync.as_ref())?;
        match self.read_record(&processing_path) {
            Ok(record) if record.envelope.event_id == pending.event_id => Ok(ClaimedEvent {
                path: processing_path,
                file_name,
                record,
            }),
            Ok(_) => {
                self.quarantine_invalid(&processing_path, "identity_mismatch")?;
                Err(SpoolError::IdentityMismatch)
            }
            Err(error) => {
                self.quarantine_invalid(&processing_path, error_class(&error))?;
                Err(error)
            }
        }
    }

    pub fn requeue(
        &self,
        mut claimed: ClaimedEvent,
        not_before: Option<String>,
    ) -> Result<(), SpoolError> {
        claimed.record.not_before = not_before;
        self.rewrite_claimed(&claimed)?;
        let pending = self.layout.spool_pending.join(&claimed.file_name);
        rename_durable(&claimed.path, &pending, self.sync.as_ref())?;
        Ok(())
    }

    pub fn commit(&self, claimed: ClaimedEvent) -> Result<(), SpoolError> {
        self.verify_claimed(&claimed)?;
        remove_durable(&claimed.path, self.sync.as_ref())?;
        Ok(())
    }

    pub fn fail(
        &self,
        mut claimed: ClaimedEvent,
        error_class: &str,
        not_before: Option<String>,
    ) -> Result<FailureDisposition, SpoolError> {
        self.verify_claimed(&claimed)?;
        claimed.record.attempts = claimed
            .record
            .attempts
            .checked_add(1)
            .ok_or(SpoolError::AttemptOverflow)?;
        claimed.record.not_before = not_before;
        claimed.record.last_error_class = Some(sanitize_error_class(error_class));
        self.rewrite_claimed(&claimed)?;
        if claimed.record.attempts >= self.limits.max_attempts {
            let failed = self.layout.spool_failed.join(&claimed.file_name);
            rename_durable(&claimed.path, &failed, self.sync.as_ref())?;
            Ok(FailureDisposition::Quarantined(claimed.record.attempts))
        } else {
            let pending = self.layout.spool_pending.join(&claimed.file_name);
            rename_durable(&claimed.path, &pending, self.sync.as_ref())?;
            Ok(FailureDisposition::Requeued(claimed.record.attempts))
        }
    }

    pub fn recover_processing<F>(&self, is_committed: F) -> Result<RecoveryReport, SpoolError>
    where
        F: Fn(&str) -> Result<bool, SpoolError>,
    {
        self.layout.ensure_private()?;
        let mut report = RecoveryReport::default();
        for entry in fs::read_dir(&self.layout.spool_processing)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let path = entry.path();
            if entry.file_name().to_string_lossy().starts_with(".tmp-") {
                continue;
            }
            let record = match self.read_record(&path) {
                Ok(record) => record,
                Err(error) => {
                    self.quarantine_invalid(&path, error_class(&error))?;
                    report.invalid_quarantined += 1;
                    continue;
                }
            };
            if is_committed(&record.envelope.event_id)? {
                remove_durable(&path, self.sync.as_ref())?;
                report.committed_removed += 1;
            } else {
                let file_name = path.file_name().ok_or(SpoolError::InvalidPath)?;
                let pending = self.layout.spool_pending.join(file_name);
                rename_durable(&path, &pending, self.sync.as_ref())?;
                report.requeued += 1;
            }
        }
        Ok(report)
    }

    pub fn depths(&self) -> Result<SpoolDepths, SpoolError> {
        self.layout.ensure_private()?;
        let (pending, pending_bytes) = tree_counts(&self.layout.spool_pending)?;
        let (processing, processing_bytes) = tree_counts(&self.layout.spool_processing)?;
        let (failed, failed_bytes) = tree_counts(&self.layout.spool_failed)?;
        Ok(SpoolDepths {
            pending,
            processing,
            failed,
            bytes: pending_bytes
                .saturating_add(processing_bytes)
                .saturating_add(failed_bytes),
        })
    }

    pub(crate) fn quarantine_superseded(
        &self,
        dataset: &str,
        maximum_generation: u64,
    ) -> Result<usize, SpoolError> {
        let mut quarantined = 0;
        for source in [&self.layout.spool_pending, &self.layout.spool_processing] {
            for entry in fs::read_dir(source)? {
                let entry = entry?;
                if !entry.file_type()?.is_file()
                    || entry.file_name().to_string_lossy().starts_with(".tmp-")
                {
                    continue;
                }
                let path = entry.path();
                let record = match self.read_record(&path) {
                    Ok(record) => record,
                    Err(error) => {
                        self.quarantine_invalid(&path, error_class(&error))?;
                        continue;
                    }
                };
                if record.envelope.dataset != dataset
                    || record.envelope.dataset_generation > maximum_generation
                {
                    continue;
                }
                let destination_directory = self
                    .layout
                    .spool_failed
                    .join("superseded")
                    .join(format!("generation-{}", record.envelope.dataset_generation));
                ensure_private_directory(&destination_directory)?;
                let destination = destination_directory.join(entry.file_name());
                rename_durable(&path, &destination, self.sync.as_ref())?;
                quarantined += 1;
            }
        }
        Ok(quarantined)
    }

    fn rewrite_claimed(&self, claimed: &ClaimedEvent) -> Result<(), SpoolError> {
        self.verify_claimed(claimed)?;
        let contents = serde_json::to_vec(&claimed.record)?;
        if contents.len() as u64 > MAX_EVENT_FILE_BYTES {
            return Err(SpoolError::EventTooLarge);
        }
        write_atomic(
            &claimed.path,
            &contents,
            ReplaceMode::Replace,
            self.sync.as_ref(),
        )?;
        Ok(())
    }

    fn verify_claimed(&self, claimed: &ClaimedEvent) -> Result<(), SpoolError> {
        if claimed.path.parent() != Some(self.layout.spool_processing.as_path()) {
            return Err(SpoolError::InvalidPath);
        }
        let parsed = parse_event_file(&claimed.path).ok_or(SpoolError::InvalidPath)?;
        if parsed.event_id != claimed.record.envelope.event_id {
            return Err(SpoolError::IdentityMismatch);
        }
        Ok(())
    }

    fn read_record(&self, path: &Path) -> Result<SpoolRecord, SpoolError> {
        if fs::metadata(path)?.len() > MAX_EVENT_FILE_BYTES {
            return Err(SpoolError::EventTooLarge);
        }
        let bytes = fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn spool_bytes(&self) -> Result<u64, SpoolError> {
        Ok(self.depths()?.bytes)
    }

    fn quarantine_invalid(&self, path: &Path, class: &str) -> Result<(), SpoolError> {
        let directory = self.layout.spool_failed.join("invalid");
        ensure_private_directory(&directory)?;
        let file_name = path.file_name().ok_or(SpoolError::InvalidPath)?;
        let mut destination = directory.join(file_name);
        if destination.exists() {
            destination = directory.join(format!(
                "{}.invalid-{}",
                file_name.to_string_lossy(),
                std::process::id()
            ));
        }
        rename_durable(path, &destination, self.sync.as_ref())?;
        let status = serde_json::to_vec(&json!({
            "error_class": sanitize_error_class(class),
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))?;
        write_atomic(
            &self.layout.status.join("spool-last-error.json"),
            &status,
            ReplaceMode::Replace,
            self.sync.as_ref(),
        )?;
        Ok(())
    }
}

fn source_unix_nanos(timestamp: &str) -> Result<i64, SpoolError> {
    let parsed =
        DateTime::parse_from_rfc3339(timestamp).map_err(|_| SpoolError::InvalidTimestamp)?;
    let nanos = parsed
        .timestamp_nanos_opt()
        .ok_or(SpoolError::InvalidTimestamp)?;
    if nanos < 0 {
        return Err(SpoolError::InvalidTimestamp);
    }
    Ok(nanos)
}

fn event_file_name(priority: Priority, source_unix_nanos: i64, event_id: &str) -> String {
    format!(
        "{}-{source_unix_nanos:020}-{event_id}.json",
        priority.as_str()
    )
}

fn parse_event_file(path: &Path) -> Option<SpoolFile> {
    let name = path.file_name()?.to_str()?;
    let name = name.strip_suffix(".json")?;
    let mut fields = name.splitn(3, '-');
    let priority = Priority::parse(fields.next()?)?;
    let source_unix_nanos = fields.next()?.parse::<i64>().ok()?;
    let event_id = fields.next()?.to_owned();
    if event_id.len() != 64
        || !event_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return None;
    }
    Some(SpoolFile {
        path: path.to_path_buf(),
        priority,
        source_unix_nanos,
        event_id,
    })
}

fn degraded_envelope(envelope: &EventEnvelope) -> EventEnvelope {
    let mut degraded = envelope.clone();
    degraded.payload = json!({
        "omitted": "[OMITTED: SPOOL HIGH WATER]",
        "event": envelope.event.as_str()
    });
    degraded.capture.capture_degraded = true;
    degraded
}

fn sanitize_error_class(value: &str) -> String {
    let token = value
        .split(|character: char| character == ':' || character.is_whitespace())
        .next()
        .unwrap_or_default();
    let sanitized: String = token
        .chars()
        .filter(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
        .take(64)
        .collect();
    if sanitized.is_empty() {
        "unknown".to_owned()
    } else {
        sanitized
    }
}

fn error_class(error: &SpoolError) -> &'static str {
    match error {
        SpoolError::Io(_) => "io",
        SpoolError::Atomic(_) => "atomic_io",
        SpoolError::Layout(_) => "layout",
        SpoolError::Json(_) => "invalid_json",
        SpoolError::InvalidTimestamp => "invalid_timestamp",
        SpoolError::InvalidPath => "invalid_path",
        SpoolError::EventTooLarge => "event_too_large",
        SpoolError::IdentityMismatch => "identity_mismatch",
        SpoolError::AttemptOverflow => "attempt_overflow",
    }
}

fn tree_counts(root: &Path) -> Result<(usize, u64), io::Error> {
    let mut files = 0usize;
    let mut bytes = 0u64;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            let (nested_files, nested_bytes) = tree_counts(&entry.path())?;
            files = files.saturating_add(nested_files);
            bytes = bytes.saturating_add(nested_bytes);
        } else if metadata.is_file() {
            files = files.saturating_add(1);
            bytes = bytes.saturating_add(metadata.len());
        }
    }
    Ok((files, bytes))
}
