//! Core algorithms — pure logic, no I/O.
//!
//! Everything in `core::*` is platform-independent. Storage and search
//! indices live in `store::*` and can be swapped (SQLite for native,
//! IndexedDB for WASM).

pub mod consolidation;
pub mod decay;
pub mod hdc;
pub mod hippocampus;
pub mod neocortex;
pub mod ner;
pub mod prediction;
pub mod recall;

#[cfg(feature = "embeddings")]
pub mod embeddings;
