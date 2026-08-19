//! Command-line surface for the Cognee agent.

use clap::{Parser, Subcommand};
use thiserror::Error;

#[derive(Debug, Parser)]
#[command(name = "cognee-agent", about = "Cognee MCP agent")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Mcp,
    Hook,
    Drain,
    Doctor,
    Recover,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("{0} is not available in this build")]
    Unavailable(&'static str),
}

/// Dispatch one agent command.
///
/// The command surface is intentionally established before command runtimes
/// are introduced; each command reports that it is unavailable to the caller.
pub fn run(cli: Cli) -> Result<(), AgentError> {
    let command = match cli.command {
        Command::Mcp => "mcp",
        Command::Hook => "hook",
        Command::Drain => "drain",
        Command::Doctor => "doctor",
        Command::Recover => "recover",
    };

    Err(AgentError::Unavailable(command))
}
