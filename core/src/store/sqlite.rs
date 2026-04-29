//! SQLite-backed Store.
//!
//! Schema:
//!
//! ```sql
//! CREATE TABLE memories (
//!   id            TEXT PRIMARY KEY,        -- UUID v4
//!   text          TEXT NOT NULL,
//!   tags          TEXT NOT NULL,           -- JSON array
//!   kind          TEXT NOT NULL,           -- decision | fact | event | preference
//!   scope_key     TEXT NOT NULL,           -- "agent|user|session"
//!   importance    REAL NOT NULL,
//!   created_at    INTEGER NOT NULL,        -- unix seconds
//!   last_accessed INTEGER NOT NULL,
//!   access_count  INTEGER NOT NULL,
//!   supersedes    TEXT,
//!   superseded_by TEXT,
//!   token_count   INTEGER NOT NULL,
//!   version       INTEGER NOT NULL
//! );
//! CREATE INDEX idx_scope ON memories(scope_key);
//! CREATE INDEX idx_kind ON memories(kind);
//! CREATE INDEX idx_active ON memories(superseded_by) WHERE superseded_by IS NULL;
//!
//! CREATE TABLE edges (
//!   from_id  TEXT NOT NULL,
//!   to_id    TEXT NOT NULL,
//!   kind     TEXT NOT NULL,
//!   PRIMARY KEY (from_id, to_id, kind)
//! );
//! ```

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{params, Connection};
use uuid::Uuid;

use crate::node::{MemoryEdge, MemoryKind, MemoryNode};
use crate::scope::Scope;
use crate::store::{Store, StoreStats};

/// SQLite-backed persistent store. Thread-safe via interior `Mutex`.
pub struct SqliteStore {
    conn: Mutex<Connection>,
}

impl SqliteStore {
    /// Open or create a SQLite store at the given path. Pass `:memory:`
    /// for an ephemeral in-memory database (used by tests).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let conn = if path.to_string_lossy() == ":memory:" {
            Connection::open_in_memory()?
        } else {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).ok();
                }
            }
            Connection::open(path)?
        };

        // Apply schema
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                text          TEXT NOT NULL,
                tags          TEXT NOT NULL,
                kind          TEXT NOT NULL,
                scope_key     TEXT NOT NULL,
                importance    REAL NOT NULL,
                created_at    INTEGER NOT NULL,
                last_accessed INTEGER NOT NULL,
                access_count  INTEGER NOT NULL,
                supersedes    TEXT,
                superseded_by TEXT,
                token_count   INTEGER NOT NULL,
                version       INTEGER NOT NULL DEFAULT 1,
                tag_sources   TEXT NOT NULL DEFAULT '[]',
                attributes    TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_scope ON memories(scope_key);
            CREATE INDEX IF NOT EXISTS idx_kind  ON memories(kind);

            CREATE TABLE IF NOT EXISTS edges (
                from_id  TEXT NOT NULL,
                to_id    TEXT NOT NULL,
                kind     TEXT NOT NULL,
                PRIMARY KEY (from_id, to_id, kind)
            );
            CREATE INDEX IF NOT EXISTS idx_edge_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edge_to   ON edges(to_id);
            "#,
        )
        .context("Failed to create Smriti schema")?;

        // Forward migration: add tag_sources to legacy databases that
        // were created before this column existed. Safe to run repeatedly:
        // we ignore the error if the column is already present.
        let _ = conn.execute(
            "ALTER TABLE memories ADD COLUMN tag_sources TEXT NOT NULL DEFAULT '[]'",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE memories ADD COLUMN attributes TEXT NOT NULL DEFAULT '{}'",
            [],
        );

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }
}

fn ts(dt: DateTime<Utc>) -> i64 {
    dt.timestamp()
}

fn from_ts(s: i64) -> DateTime<Utc> {
    Utc.timestamp_opt(s, 0).single().unwrap_or_else(Utc::now)
}

fn edge_kind_to_str(e: MemoryEdge) -> &'static str {
    match e {
        MemoryEdge::RelatesTo => "relates_to",
        MemoryEdge::Contradicts => "contradicts",
        MemoryEdge::Supports => "supports",
        MemoryEdge::DerivedFrom => "derived_from",
        MemoryEdge::Supersedes => "supersedes",
        MemoryEdge::Before => "before",
        MemoryEdge::After => "after",
        MemoryEdge::CausedBy => "caused_by",
    }
}

fn edge_kind_from_str(s: &str) -> Option<MemoryEdge> {
    MemoryEdge::parse(s)
}

impl Store for SqliteStore {
    fn upsert(&self, node: &MemoryNode) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let tags_json = serde_json::to_string(&node.tags)?;
        // Pad tag_sources to match tags length on the way out — defensive
        // against any caller that mutated `tags` without updating sources.
        let mut sources = node.tag_sources.clone();
        sources.resize(node.tags.len(), crate::node::TagSource::User);
        let sources_json = serde_json::to_string(&sources)?;
        let attributes_json = serde_json::to_string(&node.attributes)?;
        conn.execute(
            r#"INSERT OR REPLACE INTO memories
               (id, text, tags, kind, scope_key, importance, created_at,
                last_accessed, access_count, supersedes, superseded_by,
                token_count, version, tag_sources, attributes)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)"#,
            params![
                node.id.to_string(),
                node.text,
                tags_json,
                node.kind.to_string(),
                node.scope.to_key(),
                node.importance as f64,
                ts(node.created_at),
                ts(node.last_accessed_at),
                node.access_count as i64,
                node.supersedes.map(|u| u.to_string()),
                node.superseded_by.map(|u| u.to_string()),
                node.token_count as i64,
                node.version as i64,
                sources_json,
                attributes_json,
            ],
        )?;
        Ok(())
    }

    fn link(&self, from: Uuid, to: Uuid, edge: MemoryEdge) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id, kind) VALUES (?1, ?2, ?3)",
            params![from.to_string(), to.to_string(), edge_kind_to_str(edge)],
        )?;
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<MemoryNode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            r#"SELECT id, text, tags, kind, scope_key, importance,
                      created_at, last_accessed, access_count, supersedes,
                      superseded_by, token_count, version, tag_sources, attributes
               FROM memories"#,
        )?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let text: String = row.get(1)?;
            let tags_json: String = row.get(2)?;
            let kind: String = row.get(3)?;
            let scope_key: String = row.get(4)?;
            let importance: f64 = row.get(5)?;
            let created_at: i64 = row.get(6)?;
            let last_accessed: i64 = row.get(7)?;
            let access_count: i64 = row.get(8)?;
            let supersedes: Option<String> = row.get(9)?;
            let superseded_by: Option<String> = row.get(10)?;
            let token_count: i64 = row.get(11)?;
            let version: i64 = row.get(12)?;
            let tag_sources_json: String = row.get(13)?;
            let attributes_json: String = row.get(14)?;

            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            let attributes: std::collections::HashMap<String, crate::node::AttributeValue> = 
                serde_json::from_str(&attributes_json).unwrap_or_default();
            let mut tag_sources: Vec<crate::node::TagSource> =
                serde_json::from_str(&tag_sources_json).unwrap_or_default();
            // Legacy rows (and explicit defaults) come back empty —
            // fill with `User` so existing data behaves as before.
            tag_sources.resize(tags.len(), crate::node::TagSource::User);
            let scope = Scope::from_key(&scope_key).unwrap_or_default();

            Ok(MemoryNode {
                id: Uuid::parse_str(&id).unwrap_or_else(|_| Uuid::new_v4()),
                text,
                tags,
                tag_sources,
                kind: MemoryKind::parse(&kind),
                salience: crate::node::Salience::Routine,
                scope,
                importance: importance as f32,
                created_at: from_ts(created_at),
                last_accessed_at: from_ts(last_accessed),
                access_count: access_count as u32,
                supersedes: supersedes.and_then(|s| Uuid::parse_str(&s).ok()),
                superseded_by: superseded_by.and_then(|s| Uuid::parse_str(&s).ok()),
                token_count: token_count as usize,
                version: version as u64,
                attributes,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    fn load_edges(&self) -> Result<Vec<(Uuid, Uuid, MemoryEdge)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT from_id, to_id, kind FROM edges")?;
        let rows = stmt.query_map([], |row| {
            let f: String = row.get(0)?;
            let t: String = row.get(1)?;
            let k: String = row.get(2)?;
            Ok((f, t, k))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (f, t, k) = r?;
            if let (Ok(fu), Ok(tu), Some(ek)) = (
                Uuid::parse_str(&f),
                Uuid::parse_str(&t),
                edge_kind_from_str(&k),
            ) {
                out.push((fu, tu, ek));
            }
        }
        Ok(out)
    }

    fn supersede(&self, old_id: Uuid, new_id: Uuid) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE memories SET superseded_by = ?1 WHERE id = ?2",
            params![new_id.to_string(), old_id.to_string()],
        )?;
        conn.execute(
            "UPDATE memories SET supersedes = ?1 WHERE id = ?2",
            params![old_id.to_string(), new_id.to_string()],
        )?;
        conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id, kind) VALUES (?1, ?2, ?3)",
            params![
                new_id.to_string(),
                old_id.to_string(),
                edge_kind_to_str(MemoryEdge::Supersedes)
            ],
        )?;
        Ok(())
    }

    fn delete(&self, id: Uuid) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id.to_string()],
        )?;
        conn.execute(
            "DELETE FROM edges WHERE from_id = ?1 OR to_id = ?1",
            params![id.to_string()],
        )?;
        Ok(())
    }

    fn touch(&self, id: Uuid) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            r#"UPDATE memories
               SET access_count = access_count + 1,
                   last_accessed = ?1
               WHERE id = ?2"#,
            params![Utc::now().timestamp(), id.to_string()],
        )?;
        Ok(())
    }

    fn stats(&self) -> Result<StoreStats> {
        let conn = self.conn.lock().unwrap();
        let total: i64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;
        let superseded: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL",
            [],
            |r| r.get(0),
        )?;
        let edges: i64 = conn.query_row("SELECT COUNT(*) FROM edges", [], |r| r.get(0))?;
        let tokens: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(token_count), 0) FROM memories",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        Ok(StoreStats {
            total_memories: total as usize,
            active_memories: (total - superseded).max(0) as usize,
            superseded_memories: superseded as usize,
            total_edges: edges as usize,
            total_tokens: tokens as usize,
        })
    }

    fn load_for_scope(&self, _scope: &Scope) -> Result<Vec<MemoryNode>> {
        // For v1, scope filtering happens in-memory after load — the
        // dataset is small enough. Future: push the filter into SQL.
        self.load_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::MemoryKind;

    #[test]
    fn roundtrip_memory() {
        let store = SqliteStore::open(":memory:").unwrap();
        let mut node = MemoryNode::new("hello world", MemoryKind::Fact, Scope::default());
        node.tags = vec!["test".to_string()];
        node.importance = 0.7;
        store.upsert(&node).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].text, "hello world");
        assert_eq!(loaded[0].tags, vec!["test"]);
        assert_eq!(loaded[0].kind, MemoryKind::Fact);
        assert!((loaded[0].importance - 0.7).abs() < 1e-6);
    }

    #[test]
    fn touch_increments_access() {
        let store = SqliteStore::open(":memory:").unwrap();
        let node = MemoryNode::new("x", MemoryKind::Fact, Scope::default());
        store.upsert(&node).unwrap();
        store.touch(node.id).unwrap();
        store.touch(node.id).unwrap();
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded[0].access_count, 2);
    }

    #[test]
    fn attributes_round_trip_through_sqlite() {
        use crate::node::AttributeValue;
        let store = SqliteStore::open(":memory:").unwrap();
        let mut node = MemoryNode::new("priced item", MemoryKind::Event, Scope::default());
        node.attributes.insert(
            "price".to_string(),
            AttributeValue::Number(75.5),
        );
        node.attributes.insert(
            "location".to_string(),
            AttributeValue::Text("Seattle".to_string()),
        );
        node.attributes.insert(
            "tags".to_string(),
            AttributeValue::List(vec![
                AttributeValue::Text("a".to_string()),
                AttributeValue::Text("b".to_string()),
            ]),
        );
        store.upsert(&node).unwrap();

        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        let got = &loaded[0];
        assert_eq!(got.attributes.len(), 3);
        assert_eq!(
            got.attributes.get("price"),
            Some(&AttributeValue::Number(75.5))
        );
        assert_eq!(
            got.attributes.get("location"),
            Some(&AttributeValue::Text("Seattle".to_string()))
        );
        match got.attributes.get("tags") {
            Some(AttributeValue::List(items)) => assert_eq!(items.len(), 2),
            other => panic!("expected list, got {other:?}"),
        }
    }

    #[test]
    fn legacy_row_without_attributes_loads_with_empty_map() {
        // The forward migration adds the attributes column with a
        // default of '{}'. A row inserted before the migration (or in a
        // clean DB without attributes) should round-trip with an empty
        // attributes HashMap, not a parse error.
        let store = SqliteStore::open(":memory:").unwrap();
        let node = MemoryNode::new("legacy", MemoryKind::Fact, Scope::default());
        // Don't set any attributes — simulate legacy ingest path.
        store.upsert(&node).unwrap();
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(
            loaded[0].attributes.is_empty(),
            "expected empty attributes on legacy load"
        );
    }

    #[test]
    fn supersede_marks_old_and_new() {
        let store = SqliteStore::open(":memory:").unwrap();
        let old = MemoryNode::new("old", MemoryKind::Fact, Scope::default());
        let new = MemoryNode::new("new", MemoryKind::Fact, Scope::default());
        store.upsert(&old).unwrap();
        store.upsert(&new).unwrap();
        store.supersede(old.id, new.id).unwrap();

        let loaded = store.load_all().unwrap();
        let old_loaded = loaded.iter().find(|n| n.id == old.id).unwrap();
        let new_loaded = loaded.iter().find(|n| n.id == new.id).unwrap();
        assert_eq!(old_loaded.superseded_by, Some(new.id));
        assert_eq!(new_loaded.supersedes, Some(old.id));
    }

    #[test]
    fn edges_roundtrip() {
        let store = SqliteStore::open(":memory:").unwrap();
        let a = MemoryNode::new("a", MemoryKind::Fact, Scope::default());
        let b = MemoryNode::new("b", MemoryKind::Fact, Scope::default());
        store.upsert(&a).unwrap();
        store.upsert(&b).unwrap();
        store.link(a.id, b.id, MemoryEdge::RelatesTo).unwrap();

        let edges = store.load_edges().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].2, MemoryEdge::RelatesTo);
    }
}
