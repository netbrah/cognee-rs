//! JSON-RPC 2.0 message handling for MCP stdio (no I/O).

use serde_json::{Value, json};

const PROTOCOL_VERSION: &str = "2024-11-05";
const SERVER_NAME: &str = "cognee-agent";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Handle one newline-stripped JSON-RPC message.
///
/// Returns `None` for notifications (no `id`).
pub fn handle_message(line: &str) -> Option<String> {
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            return Some(
                json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {"code": -32700, "message": "Parse error"}
                })
                .to_string(),
            );
        }
    };

    let method = msg.get("method").and_then(Value::as_str)?;
    let id = msg.get("id").cloned();

    if method == "initialize" {
        let id = id?;
        return Some(
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": { "tools": {} },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION
                    }
                }
            })
            .to_string(),
        );
    }

    if method == "ping" {
        let id = id?;
        return Some(json!({"jsonrpc": "2.0", "id": id, "result": {}}).to_string());
    }

    let id = id?;
    Some(
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": "Method not found"}
        })
        .to_string(),
    )
}
