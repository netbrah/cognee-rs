//! Command-line surface for the Cognee agent.

use clap::{Parser, Subcommand};

pub use crate::error::AgentError;

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

/// Dispatch one agent command.
///
/// Hook capture is available in runtime builds. The remaining command
/// runtimes retain their stable placeholder errors until their planned tasks.
pub fn run(cli: Cli) -> Result<(), AgentError> {
    #[cfg(feature = "runtime")]
    if matches!(cli.command, Command::Hook) {
        return crate::hook::run_hook(std::io::stdin().lock(), std::io::stdout().lock())
            .map_err(|_| AgentError::Unavailable("hook output"));
    }

    let command = match cli.command {
        Command::Mcp => "mcp",
        Command::Hook => "hook",
        Command::Drain => "drain",
        Command::Doctor => "doctor",
        Command::Recover => "recover",
    };

    Err(AgentError::Unavailable(command))
}
