use clap::Parser;
use cognee_mcp::cli::Cli;

#[test]
fn parses_all_five_top_level_commands() {
    for command in ["mcp", "hook", "drain", "doctor", "recover"] {
        assert!(
            Cli::try_parse_from(["cognee-agent", command]).is_ok(),
            "{command}"
        );
    }
}
