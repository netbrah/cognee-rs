//! Curated, fleet-readable reference-memory contracts.
//!
//! Reference memory is deliberately independent from the private user state
//! used by hooks and the existing memory tools.  This module contains only the
//! stable configuration, layout, and administrator command surface; storage
//! and recall are layered on in the sibling modules.

mod config;
mod layout;

use std::path::PathBuf;

use clap::{Args, Subcommand};

pub use config::{REFERENCE_DATASET, ReferenceConfig, ReferenceLimits};
pub use layout::ReferenceLayout;

pub use crate::error::ReferenceError;

pub const DEFAULT_WAIT_SECONDS: u64 = 1_800;
pub const MAX_WAIT_SECONDS: u64 = 7_200;

#[derive(Debug, Subcommand)]
pub enum ReferenceCommand {
    Remember(ReferenceRememberArgs),
    Publish,
    Doctor {
        #[arg(long)]
        json: bool,
    },
    Recover {
        #[arg(long)]
        adopt_orphans: bool,
    },
}

#[derive(Debug, Args)]
pub struct ReferenceRememberArgs {
    #[arg(short = 'f', long = "file", action = clap::ArgAction::Append)]
    pub files: Vec<PathBuf>,
    #[arg(long, conflicts_with = "files")]
    pub source_id: Option<String>,
    #[arg(long, conflicts_with = "files")]
    pub label: Option<String>,
    #[arg(long)]
    pub wait_cognified: bool,
    #[arg(
        long,
        default_value_t = DEFAULT_WAIT_SECONDS,
        value_parser = clap::value_parser!(u64).range(1..=MAX_WAIT_SECONDS)
    )]
    pub wait_seconds: u64,
}

impl ReferenceRememberArgs {
    pub fn uses_stdin(&self) -> bool {
        self.files.is_empty()
    }
}
