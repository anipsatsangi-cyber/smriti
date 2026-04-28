//! Tantivy full-text index for fast keyword search over memories.
//!
//! Mirrors the pattern in codegraph-index/src/schema.rs and searcher.rs
//! but simplified for single-field (text) memories.

use anyhow::{Context, Result};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value, STORED, TEXT};
use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument};
use uuid::Uuid;

/// Wrapper around a Tantivy index for memory text search.
pub struct MemoryIndex {
    index: Index,
    schema: Schema,
    writer: Option<IndexWriter>,
}

impl MemoryIndex {
    /// Create or open a tantivy index at `dir`.
    pub fn open(dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let mut builder = Schema::builder();
        builder.add_text_field("id", STORED);
        builder.add_text_field("text", TEXT | STORED);
        builder.add_text_field("tags", TEXT | STORED);
        let schema = builder.build();

        let index = if dir.join("meta.json").exists() {
            Index::open_in_dir(dir).context("Failed to open existing memory index")?
        } else {
            Index::create_in_dir(dir, schema.clone())
                .context("Failed to create memory index")?
        };

        Ok(Self { index, schema, writer: None })
    }

    /// Open an in-RAM index (for tests / ephemeral use).
    pub fn open_in_ram() -> Result<Self> {
        let mut builder = Schema::builder();
        builder.add_text_field("id", STORED);
        builder.add_text_field("text", TEXT | STORED);
        builder.add_text_field("tags", TEXT | STORED);
        let schema = builder.build();
        let index = Index::create_in_ram(schema.clone());
        Ok(Self { index, schema, writer: None })
    }

    fn writer(&mut self) -> Result<&mut IndexWriter> {
        if self.writer.is_none() {
            self.writer = Some(
                self.index
                    .writer(16_000_000)
                    .context("Failed to create index writer")?,
            );
        }
        Ok(self.writer.as_mut().unwrap())
    }

    /// Add a memory to the index. Call `commit()` to persist.
    pub fn add(&mut self, id: &Uuid, text: &str, tags: &[String]) -> Result<()> {
        let id_field = self.schema.get_field("id").unwrap();
        let text_field = self.schema.get_field("text").unwrap();
        let tags_field = self.schema.get_field("tags").unwrap();

        let mut doc = TantivyDocument::default();
        doc.add_text(id_field, id.to_string());
        doc.add_text(text_field, text);
        doc.add_text(tags_field, tags.join(" "));

        self.writer()?.add_document(doc)?;
        Ok(())
    }

    /// Remove a memory from the index by its Uuid.
    pub fn remove(&mut self, id: &Uuid) -> Result<()> {
        let id_field = self.schema.get_field("id").unwrap();
        let term = tantivy::Term::from_field_text(id_field, &id.to_string());
        self.writer()?.delete_term(term);
        Ok(())
    }

    /// Commit pending index writes.
    pub fn commit(&mut self) -> Result<()> {
        if let Some(w) = self.writer.as_mut() {
            w.commit()?;
        }
        Ok(())
    }

    /// Search for memories matching `query_str`. Returns up to `limit` Uuid strings.
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<Uuid>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to build index reader")?;

        let searcher = reader.searcher();
        let text_field = self.schema.get_field("text").unwrap();
        let tags_field = self.schema.get_field("tags").unwrap();
        let id_field = self.schema.get_field("id").unwrap();

        let parser = QueryParser::for_index(&self.index, vec![text_field, tags_field]);
        let query = match parser.parse_query(query_str) {
            Ok(q) => q,
            Err(_) => {
                // Fallback: escape the query and try again
                let escaped = query_str
                    .chars()
                    .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                    .collect::<String>();
                parser.parse_query(&escaped)?
            }
        };

        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        let mut ids = Vec::new();
        for (_score, addr) in top_docs {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id_val) = doc.get_first(id_field) {
                if let Some(id_str) = id_val.as_str() {
                    if let Ok(uuid) = Uuid::parse_str(id_str) {
                        ids.push(uuid);
                    }
                }
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_search_commit() {
        let mut idx = MemoryIndex::open_in_ram().unwrap();
        let id = Uuid::new_v4();
        idx.add(&id, "The auth module uses JWT RS256 tokens", &["auth".to_string()])
            .unwrap();
        idx.commit().unwrap();

        let results = idx.search("JWT", 10).unwrap();
        assert!(results.contains(&id));
    }

    #[test]
    fn test_remove() {
        let mut idx = MemoryIndex::open_in_ram().unwrap();
        let id = Uuid::new_v4();
        idx.add(&id, "Database uses PostgreSQL", &[]).unwrap();
        idx.commit().unwrap();

        idx.remove(&id).unwrap();
        idx.commit().unwrap();

        let results = idx.search("PostgreSQL", 10).unwrap();
        assert!(!results.contains(&id));
    }
}
