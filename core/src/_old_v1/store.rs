//! SQLite persistence for the memory engine.
//!
//! Schema mirrors the arena-lab pattern (rusqlite + bundled SQLite).
//! Two tables: `memories` and `memory_edges`.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use uuid::Uuid;

use crate::graph::{MemoryEdge, MemoryNode};

/// Thin wrapper around a rusqlite Connection.
pub struct MemoryStore {
    conn: Connection,
}

impl MemoryStore {
    /// Open (or create) a SQLite database at `path`.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open memory store at {}", path.display()))?;
        let store = Self { conn };
        store.create_tables()?;
        Ok(store)
    }

    /// Open an in-memory database (useful for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let store = Self { conn };
        store.create_tables()?;
        Ok(store)
    }

    fn create_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT    PRIMARY KEY,
                text        TEXT    NOT NULL,
                tags        TEXT    NOT NULL DEFAULT '[]',
                importance  REAL    NOT NULL DEFAULT 0.5,
                created_at  TEXT    NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS memory_edges (
                from_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                to_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                kind    TEXT NOT NULL,
                PRIMARY KEY (from_id, to_id, kind)
            );
            ",
        )?;
        Ok(())
    }

    // ── Nodes ──────────────────────────────────────────────────────────────────

    /// Persist a new memory node.
    pub fn insert_node(&self, node: &MemoryNode) -> Result<()> {
        let tags_json = serde_json::to_string(&node.tags)?;
        self.conn.execute(
            "INSERT OR REPLACE INTO memories (id, text, tags, importance, created_at, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                node.id.to_string(),
                node.text,
                tags_json,
                node.importance,
                node.created_at,
                node.token_count as i64,
            ],
        )?;
        Ok(())
    }

    /// Delete a memory by id (cascades to edges).
    pub fn delete_node(&self, id: &Uuid) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id.to_string()],
        )?;
        Ok(rows > 0)
    }

    /// Load all memories from the database.
    pub fn load_all_nodes(&self) -> Result<Vec<MemoryNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, text, tags, importance, created_at, token_count FROM memories ORDER BY created_at",
        )?;
        let nodes = stmt
            .query_map([], |row| {
                let id_str: String = row.get(0)?;
                let tags_json: String = row.get(2)?;
                Ok((id_str, row.get(1)?, tags_json, row.get(3)?, row.get(4)?, row.get::<_, i64>(5)?))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id_str, text, tags_json, importance, created_at, token_count)| {
                let id = Uuid::parse_str(&id_str).ok()?;
                let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                Some(MemoryNode {
                    id,
                    text,
                    tags,
                    importance,
                    created_at,
                    token_count: token_count as usize,
                })
            })
            .collect();
        Ok(nodes)
    }

    // ── Edges ──────────────────────────────────────────────────────────────────

    /// Persist an edge between two memories.
    pub fn insert_edge(&self, from: &Uuid, to: &Uuid, kind: MemoryEdge) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_edges (from_id, to_id, kind) VALUES (?1, ?2, ?3)",
            params![from.to_string(), to.to_string(), kind.to_string()],
        )?;
        Ok(())
    }

    /// Load all edges as (from_uuid, to_uuid, kind) triples.
    pub fn load_all_edges(&self) -> Result<Vec<(Uuid, Uuid, MemoryEdge)>> {
        let mut stmt = self.conn.prepare(
            "SELECT from_id, to_id, kind FROM memory_edges",
        )?;
        let edges = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(from_str, to_str, kind_str)| {
                let from = Uuid::parse_str(&from_str).ok()?;
                let to = Uuid::parse_str(&to_str).ok()?;
                let kind = match kind_str.as_str() {
                    "contradicts" => MemoryEdge::Contradicts,
                    "supports" => MemoryEdge::Supports,
                    "derived_from" => MemoryEdge::DerivedFrom,
                    _ => MemoryEdge::RelatesTo,
                };
                Some((from, to, kind))
            })
            .collect();
        Ok(edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(text: &str) -> MemoryNode {
        MemoryNode {
            id: Uuid::new_v4(),
            text: text.to_string(),
            tags: vec!["test".to_string()],
            importance: 0.7,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            token_count: 5,
        }
    }

    #[test]
    fn test_insert_and_load() {
        let store = MemoryStore::open_in_memory().unwrap();
        let n = make_node("The auth uses JWT");
        store.insert_node(&n).unwrap();
        let nodes = store.load_all_nodes().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].text, "The auth uses JWT");
    }

    #[test]
    fn test_delete_cascades() {
        let store = MemoryStore::open_in_memory().unwrap();
        let n1 = make_node("A");
        let n2 = make_node("B");
        let id1 = n1.id;
        store.insert_node(&n1).unwrap();
        store.insert_node(&n2).unwrap();
        store.insert_edge(&id1, &n2.id, MemoryEdge::RelatesTo).unwrap();
        store.delete_node(&id1).unwrap();
        let edges = store.load_all_edges().unwrap();
        assert!(edges.is_empty(), "cascade delete should remove edges");
    }

    #[test]
    fn test_edges_round_trip() {
        let store = MemoryStore::open_in_memory().unwrap();
        let n1 = make_node("X");
        let n2 = make_node("Y");
        let id1 = n1.id;
        let id2 = n2.id;
        store.insert_node(&n1).unwrap();
        store.insert_node(&n2).unwrap();
        store.insert_edge(&id1, &id2, MemoryEdge::Contradicts).unwrap();
        let edges = store.load_all_edges().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].2, MemoryEdge::Contradicts);
    }
}
