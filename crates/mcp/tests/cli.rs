use clap::Parser;
use cognee_mcp::cli::{Cli, Command};

#[test]
fn parses_all_five_top_level_commands() {
    for command in ["mcp", "hook", "drain", "doctor", "recover"] {
        assert!(
            Cli::try_parse_from(["cognee-agent", command]).is_ok(),
            "{command}"
        );
    }
}

#[test]
fn parses_read_only_recall_diagnostic_arguments() {
    let cli = Cli::try_parse_from([
        "cognee-agent",
        "recall",
        "--query",
        "stable preferences",
        "--session-id",
        "session-123",
        "--search-type",
        "CHUNKS",
        "--top-k",
        "7",
    ])
    .expect("parse recall diagnostic");

    match cli.command {
        Command::Recall {
            query,
            session_id,
            search_type,
            top_k,
        } => {
            assert_eq!(query, "stable preferences");
            assert_eq!(session_id.as_deref(), Some("session-123"));
            assert_eq!(search_type, "CHUNKS");
            assert_eq!(top_k, 7);
        }
        command => panic!("expected recall command, got {command:?}"),
    }
}

#[test]
#[cfg(feature = "engine")]
fn drain_command_runs_an_empty_bounded_worker_without_opening_storage() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let root = temporary.path().join("cognee");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cognee-agent"))
        .arg("drain")
        .env_clear()
        .env("APEX_COGNEE_ROOT", &root)
        .env("APEX_COGNEE_PROXY_KEY", "fixture-secret")
        .env("APEX_COGNEE_LLM_PROVIDER", "openai")
        .env("APEX_COGNEE_LLM_ENDPOINT", "https://proxy.example/v1")
        .env("APEX_COGNEE_LLM_MODEL", "gpt-5.4-nano")
        .env("APEX_COGNEE_EMBEDDING_PROVIDER", "openai")
        .env(
            "APEX_COGNEE_EMBEDDING_ENDPOINT",
            "https://proxy.example/v1/embeddings",
        )
        .env("APEX_COGNEE_EMBEDDING_MODEL", "text-embedding-3-large")
        .env("APEX_COGNEE_EMBEDDING_DIMENSIONS", "3072")
        .output()
        .expect("run drain command");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    assert!(root.join("spool/pending").is_dir());
    assert!(root.join("ledger/ingestion.sqlite3").is_file());
    assert!(!root.join("locks/engine").exists());
    assert!(
        !walk_files(&root)
            .iter()
            .any(|path| { path.file_name().is_some_and(|name| name == "cognee.db") }),
        "an empty drain must not warm Cognee storage"
    );
}

#[cfg(feature = "engine")]
fn walk_files(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(path) else {
            continue;
        };
        for entry in entries {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}
