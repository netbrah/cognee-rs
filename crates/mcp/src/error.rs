//! Stable, secret-safe errors surfaced by the Cognee agent boundary.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReferenceError {
    #[error("reference root is invalid")]
    InvalidRoot,
    #[error("reference memory is unavailable")]
    Unavailable,
    #[error("reference memory model fingerprint does not match")]
    ModelMismatch,
    #[error("reference memory is read-only")]
    ReadOnly,
    #[error("reference memory contains a corrupt record")]
    CorruptRecord,
    #[error("reference memory backlog limit reached; publish before retrying")]
    BacklogLimit,
    #[error("reference memory I/O failed")]
    Io(#[source] std::io::Error),
}

impl ReferenceError {
    pub const fn class(&self) -> &'static str {
        match self {
            Self::InvalidRoot | Self::Unavailable | Self::Io(_) => "REFERENCE_UNAVAILABLE",
            Self::ModelMismatch => "REFERENCE_MODEL_MISMATCH",
            Self::ReadOnly => "REFERENCE_READ_ONLY",
            Self::CorruptRecord => "REFERENCE_CORRUPT_RECORD",
            Self::BacklogLimit => "REFERENCE_BACKLOG_LIMIT",
        }
    }

    pub const fn retryable(&self) -> bool {
        matches!(self, Self::Unavailable | Self::Io(_) | Self::BacklogLimit)
    }
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("{0} is not available in this build")]
    Unavailable(&'static str),
    #[error("memory engine operation failed: {0}")]
    Engine(&'static str),
    #[error("memory event is missing required field {0}")]
    InvalidEvent(&'static str),
    #[error("{0} exceeded the worker deadline")]
    Timeout(&'static str),
    #[error("checkpoint state operation failed: {0}")]
    Checkpoint(&'static str),
    #[error("retryable memory engine failure: {0}")]
    Retryable(&'static str),
    #[error("blocked memory engine failure: {0}")]
    Blocked(&'static str),
    #[error("{0}")]
    Diagnostic(String),
    #[cfg(feature = "runtime")]
    #[error(transparent)]
    Lease(#[from] crate::lease::LeaseError),
    #[cfg(feature = "runtime")]
    #[error(transparent)]
    Ledger(#[from] crate::ledger::LedgerError),
    #[cfg(feature = "runtime")]
    #[error("injected worker fault at {0:?}")]
    InjectedFault(crate::worker::FaultPoint),
    #[error(transparent)]
    Spool(#[from] crate::spool::SpoolError),
    #[error(transparent)]
    Reference(#[from] ReferenceError),
}

impl AgentError {
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Unavailable(_) => "unavailable",
            Self::Engine(_) => "engine",
            Self::InvalidEvent(_) => "invalid_event",
            Self::Timeout(_) => "timeout",
            Self::Checkpoint(_) => "checkpoint",
            Self::Retryable(class) | Self::Blocked(class) => class,
            Self::Diagnostic(_) => "diagnostic",
            #[cfg(feature = "runtime")]
            Self::Lease(crate::lease::LeaseError::LeaseLost) => "lease_lost",
            #[cfg(feature = "runtime")]
            Self::Lease(_) => "lease",
            #[cfg(feature = "runtime")]
            Self::Ledger(_) => "ledger",
            #[cfg(feature = "runtime")]
            Self::InjectedFault(_) => "injected_fault",
            Self::Spool(_) => "spool",
            Self::Reference(error) => error.class(),
        }
    }

    pub const fn retry_class(&self) -> Option<&'static str> {
        match self {
            Self::Retryable(class) => Some(class),
            Self::Timeout(_) => Some("timeout"),
            Self::Reference(error) if error.retryable() => Some(error.class()),
            _ => None,
        }
    }
}
