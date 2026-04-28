//! Standalone Smriti MCP server (JSON-RPC 2.0 over stdio).
//!
//! This is the integration path for Claude Code, Cursor, Windsurf, Zed,
//! and every other MCP-compatible coding agent. Configure the agent to
//! launch:
//!
//! ```text
//! smriti mcp --db ~/.smriti/global.db
//! ```
//!
//! and Smriti's seven tools become available as native LLM functions.
//!
//! Wire format follows the [Model Context Protocol] spec:
//!
//! - One JSON-RPC request per line on stdin.
//! - One JSON-RPC response per line on stdout.
//! - Implements `initialize`, `tools/list`, `tools/call` methods.
//!
//! [Model Context Protocol]: https://modelcontextprotocol.io/

#![cfg(feature = "native")]

use std::io::{self, BufRead, Write};
use std::sync::Mutex;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::node::{MemoryEdge, MemoryKind};
use crate::scope::Scope;
use crate::Smriti;

// ── Wire types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct McpRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct McpResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<McpError>,
}

#[derive(Debug, Serialize)]
struct McpError {
    code: i32,
    message: String,
}

// ── Server ──────────────────────────────────────────────────────────────

/// Run the MCP server on stdio. Blocks until stdin closes.
pub fn run(smriti: Smriti) -> Result<()> {
    let smriti = Mutex::new(smriti);
    let stdin = io::stdin();
    let stdout = io::stdout();

    eprintln!("Smriti MCP server starting on stdio (waiting for JSON-RPC requests)…");

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<McpRequest>(&line) {
            Ok(req) => handle(req, &smriti),
            Err(e) => McpResponse {
                jsonrpc: "2.0",
                id: Value::Null,
                result: None,
                error: Some(McpError {
                    code: -32700,
                    message: format!("Parse error: {}", e),
                }),
            },
        };

        let json = serde_json::to_string(&response)?;
        let mut out = stdout.lock();
        writeln!(out, "{}", json)?;
        out.flush()?;
    }

    Ok(())
}

fn handle(req: McpRequest, smriti: &Mutex<Smriti>) -> McpResponse {
    let id = req.id.clone().unwrap_or(Value::Null);
    match req.method.as_str() {
        "initialize" => McpResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "smriti",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
            error: None,
        },
        "tools/list" => McpResponse {
            jsonrpc: "2.0",
            id,
            result: Some(json!({ "tools": tool_list() })),
            error: None,
        },
        "tools/call" => match call_tool(req.params.unwrap_or(Value::Null), smriti) {
            Ok(text) => McpResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": false
                })),
                error: None,
            },
            Err(e) => McpResponse {
                jsonrpc: "2.0",
                id,
                result: Some(json!({
                    "content": [{ "type": "text", "text": format!("Error: {}", e) }],
                    "isError": true
                })),
                error: None,
            },
        },
        // MCP spec: notifications have no id and no response.
        m if m.starts_with("notifications/") => McpResponse {
            jsonrpc: "2.0",
            id,
            result: Some(Value::Null),
            error: None,
        },
        _ => McpResponse {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(McpError {
                code: -32601,
                message: format!("Method not found: {}", req.method),
            }),
        },
    }
}

// ── Tool definitions ────────────────────────────────────────────────────

fn tool_list() -> Vec<Value> {
    vec![
        json!({
            "name": "smriti_remember",
            "description": "Store a fact, decision, event, or preference in persistent memory. Memories survive across sessions and are indexed for fast keyword + graph + (optional) embedding recall. Use this whenever the agent learns something the user expects it to remember next time.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory content. Be specific: 'User prefers Rust over Python for backend services' is better than 'Rust'."
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "decision", "event", "preference"],
                        "description": "Cognitive kind of memory. fact=stable knowledge, decision=architectural choice, event=time-stamped happening, preference=user/agent style. Default: fact.",
                        "default": "fact"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tags for filtering and graph linking. Examples: ['auth', 'security'], ['user', 'style']."
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance 0.0–1.0. Affects ranking in snapshots and decay. Default: 0.5.",
                        "default": 0.5
                    }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "smriti_recall",
            "description": "Recall memories matching a query within a strict token budget. Returns the most relevant memories combining keyword search + HDC fingerprint + graph PPR + (optional) dense embeddings. Use this at session start or before any task that benefits from prior context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question. Examples: 'how does authentication work', 'what does the user prefer for code style'"
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Max tokens to return. Default: 2000. Smriti packs the highest-scoring memories that fit.",
                        "default": 2000
                    },
                    "kinds": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional kind filter. Example: ['preference'] to recall only user preferences."
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional tag hints. Used as graph seeds and as fingerprint inputs."
                    }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "smriti_forget",
            "description": "Soft-delete a memory by UUID. The memory is hidden from recall but kept in the audit trail (use smriti_supersede if you want to replace one memory with another).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "UUID of the memory to forget."
                    }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "smriti_supersede",
            "description": "Replace an old memory with a new one. The old memory is hidden from recall and the supersedes-chain is preserved as a graph edge. Use this when a fact is updated (e.g. user changed their preference, address moved, decision was reversed).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "old_id": { "type": "string", "description": "UUID of the outdated memory." },
                    "new_text": { "type": "string", "description": "Replacement memory text." },
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "decision", "event", "preference"],
                        "default": "fact"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["old_id", "new_text"]
            }
        }),
        json!({
            "name": "smriti_link",
            "description": "Create a typed edge between two existing memories (e.g. mark one as supporting or contradicting another). Strengthens the graph for future PPR-ranked recall.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "UUID of source memory." },
                    "to": { "type": "string", "description": "UUID of target memory." },
                    "edge": {
                        "type": "string",
                        "enum": ["relates_to", "contradicts", "supports", "derived_from", "supersedes", "before", "after", "caused_by"],
                        "default": "relates_to"
                    }
                },
                "required": ["from", "to"]
            }
        }),
        json!({
            "name": "smriti_consolidate",
            "description": "Force a consolidation pass — drains the hippocampus (recent buffer) into the neocortex (long-term graph). Auto-creates RelatesTo edges for memories sharing tags. Normally happens automatically; call this explicitly if you want immediate inclusion of just-stored memories in PPR-ranked recall.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "smriti_stats",
            "description": "Return aggregate stats about the memory store: total memories, active vs superseded, edges, total tokens stored, hippocampus/neocortex sizes. Useful for observability.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
    ]
}

// ── Tool dispatch ───────────────────────────────────────────────────────

fn call_tool(params: Value, smriti: &Mutex<Smriti>) -> Result<String> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing tool name"))?
        .to_string();
    let args = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));

    match name.as_str() {
        "smriti_remember" => tool_remember(args, smriti),
        "smriti_recall" => tool_recall(args, smriti),
        "smriti_forget" => tool_forget(args, smriti),
        "smriti_supersede" => tool_supersede(args, smriti),
        "smriti_link" => tool_link(args, smriti),
        "smriti_consolidate" => tool_consolidate(smriti),
        "smriti_stats" => tool_stats(smriti),
        other => Err(anyhow::anyhow!("Unknown tool: {}", other)),
    }
}

fn arg_str(args: &Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("missing required argument: {}", key))
}

fn arg_str_opt(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

fn arg_array(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

// ── Tools ───────────────────────────────────────────────────────────────

fn tool_remember(args: Value, smriti: &Mutex<Smriti>) -> Result<String> {
    let text = arg_str(&args, "text")?;
    let kind_str = arg_str_opt(&args, "kind").unwrap_or_else(|| "fact".to_string());
    let tags = arg_array(&args, "tags");
    let importance = args
        .get("importance")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;

    let mut s = smriti.lock().unwrap();
    let mut b = s
        .remember(text)
        .kind(MemoryKind::parse(&kind_str))
        .scope(Scope::default())
        .importance(importance);
    if !tags.is_empty() {
        b = b.tags(tags);
    }
    let id = b.commit()?;
    Ok(format!("Stored memory {}", id))
}

fn tool_recall(args: Value, smriti: &Mutex<Smriti>) -> Result<String> {
    let query = arg_str(&args, "query")?;
    let budget = args
        .get("budget")
        .and_then(|v| v.as_u64())
        .unwrap_or(2000) as usize;
    let kinds: Vec<MemoryKind> = arg_array(&args, "kinds")
        .iter()
        .map(|s| MemoryKind::parse(s))
        .collect();
    let tag_hints = arg_array(&args, "tags");

    let s = smriti.lock().unwrap();
    let mut b = s.recall(query.clone()).budget(budget);
    if !tag_hints.is_empty() {
        b = b.tags(tag_hints);
    }
    if !kinds.is_empty() {
        b = b.kinds(kinds);
    }
    let result = b.execute()?;

    let mut out = format!(
        "Recalled {} memories ({} / {} tokens, {} candidates considered):\n\n",
        result.hits.len(),
        result.tokens_used,
        result.tokens_budget,
        result.candidates_considered
    );

    for (i, h) in result.hits.iter().enumerate() {
        out.push_str(&format!(
            "{}. [score={:.2}] [{}] {}\n",
            i + 1,
            h.final_score,
            h.node.kind,
            h.node.text
        ));
        if !h.node.tags.is_empty() {
            out.push_str(&format!("    tags: {}\n", h.node.tags.join(", ")));
        }
        out.push_str(&format!("    id: {}\n", h.node.id));
    }
    if result.hits.is_empty() {
        out.push_str("(no matching memories — try different keywords or tags)\n");
    }
    Ok(out)
}

fn tool_forget(args: Value, smriti: &Mutex<Smriti>) -> Result<String> {
    let id = arg_str(&args, "id")?;
    let uuid = uuid::Uuid::parse_str(&id)?;
    let mut s = smriti.lock().unwrap();
    s.forget(uuid)?;
    Ok(format!("Forgot memory {}", id))
}

fn tool_supersede(args: Value, smriti: &Mutex<Smriti>) -> Result<String> {
    let old_id = arg_str(&args, "old_id")?;
    let new_text = arg_str(&args, "new_text")?;
    let kind_str = arg_str_opt(&args, "kind").unwrap_or_else(|| "fact".to_string());
    let tags = arg_array(&args, "tags");
    let old_uuid = uuid::Uuid::parse_str(&old_id)?;

    let mut s = smriti.lock().unwrap();
    let mut b = s
        .remember(new_text)
        .kind(MemoryKind::parse(&kind_str))
        .scope(Scope::default())
        .supersedes(old_uuid);
    if !tags.is_empty() {
        b = b.tags(tags);
    }
    let new_id = b.commit()?;
    Ok(format!("Superseded {} → {}", old_id, new_id))
}

fn tool_link(args: Value, smriti: &Mutex<Smriti>) -> Result<String> {
    let from = arg_str(&args, "from")?;
    let to = arg_str(&args, "to")?;
    let edge_str = arg_str_opt(&args, "edge").unwrap_or_else(|| "relates_to".to_string());
    let edge = match edge_str.as_str() {
        "contradicts" => MemoryEdge::Contradicts,
        "supports" => MemoryEdge::Supports,
        "derived_from" => MemoryEdge::DerivedFrom,
        "supersedes" => MemoryEdge::Supersedes,
        "before" => MemoryEdge::Before,
        "after" => MemoryEdge::After,
        "caused_by" => MemoryEdge::CausedBy,
        _ => MemoryEdge::RelatesTo,
    };

    let from_uuid = uuid::Uuid::parse_str(&from)?;
    let to_uuid = uuid::Uuid::parse_str(&to)?;
    let mut s = smriti.lock().unwrap();
    s.link(from_uuid, to_uuid, edge)?;
    Ok(format!("Linked {} -[{}]-> {}", from, edge_str, to))
}

fn tool_consolidate(smriti: &Mutex<Smriti>) -> Result<String> {
    let mut s = smriti.lock().unwrap();
    let report = s.consolidate()?;
    Ok(format!(
        "Consolidation: {} processed, {} promoted, {} reinforced, {} dropped, {} edges created",
        report.processed, report.promoted, report.reinforced, report.dropped, report.edges_created
    ))
}

fn tool_stats(smriti: &Mutex<Smriti>) -> Result<String> {
    let s = smriti.lock().unwrap();
    let stats = s.stats()?;
    Ok(format!(
        "Memories: {} total, {} active, {} superseded · Edges: {} · Tokens stored: {} · Hippocampus: {}/{} · Neocortex: {} nodes / {} edges",
        stats.store.total_memories,
        stats.store.active_memories,
        stats.store.superseded_memories,
        stats.store.total_edges,
        stats.store.total_tokens,
        stats.hippocampus_size,
        stats.hippocampus_capacity,
        stats.neocortex_size,
        stats.neocortex_edges
    ))
}
