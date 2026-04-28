//! Smriti — structured memory engine for AI agents.
//!
//! स्मृति · "that which is remembered, well"
//!
//! # Architecture
//!
//! Smriti implements a **dual-store memory** inspired by the Complementary
//! Learning Systems (CLS) theory of McClelland, McNaughton & O'Reilly (1995):
//!
//! - **Hippocampus** — fast, sparse, episodic buffer for recent experiences.
//!   Uses sparse binary hypervectors (Kanerva 1988) for pattern separation.
//! - **Neocortex** — slow, dense, semantic graph for consolidated knowledge.
//!   Uses petgraph + Personalized PageRank for relational recall.
//!
//! Memories flow from Hippocampus → Neocortex via a **consolidation** pass
//! (the "sleep replay" phase), filtered by an information-theoretic
//! redundancy check based on Minimum Description Length (Rissanen, Tishby).
//!
//! Composition uses **Hyperdimensional Computing** (Plate, Kanerva, Gayler):
//! `bind` and `bundle` operations let multiple facts about an entity collapse
//! into a single algebraic structure that can be queried by deconvolution.

pub mod core;
pub mod protocol;
pub mod store;

#[cfg(feature = "http")]
pub mod http;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

mod node;
mod scope;
mod smriti;

pub use core::recall::{QueryIntent, RecallHit, RecallResult, RecallTrace, RecallVerdict};
pub use node::{MemoryEdge, MemoryKind, MemoryNode, TagSource};
pub use scope::Scope;
pub use smriti::{RecallBuilder, RememberBuilder, Smriti, SmritiStats};
