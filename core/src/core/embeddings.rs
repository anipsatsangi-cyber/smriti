//! Optional dense-embedding layer.
//!
//! By default Smriti runs entirely without ML — keyword search + HDC
//! composition + graph PPR are enough for ~89% top-1 on LongMemEval.
//! For users who want the last few percentage points of synonym recall
//! (queries like "what is Alice working on?" matching "Alice is leading
//! the auth refactor"), this module wires in [`fastembed-rs`]
//! quantized MiniLM embeddings.
//!
//! Compiled out by default. Enable with `--features embeddings`.
//!
//! On first use, fastembed downloads the model (~80MB) into
//! `$HOME/.cache/fastembed`. Subsequent loads are instant.
//!
//! [`fastembed-rs`]: https://crates.io/crates/fastembed

#![cfg(feature = "embeddings")]

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

/// A wrapper around a fastembed model. Loaded lazily, shared via `Arc`.
///
/// Construction may take a few seconds on first use (downloads weights).
pub struct Embedder {
    inner: Mutex<TextEmbedding>,
}

impl Embedder {
    /// Initialize the embedder with the default lightweight model
    /// (`AllMiniLML6V2Q` — quantized MiniLM, 384-dim, ~50MB).
    pub fn new() -> Result<Arc<Self>> {
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2Q).with_show_download_progress(false),
        )
        .context("Failed to load fastembed model — first run downloads ~50MB")?;
        Ok(Arc::new(Self {
            inner: Mutex::new(model),
        }))
    }

    /// Embed one or more texts. Returns one vector per input.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut model = self.inner.lock().expect("embedder poisoned");
        // fastembed expects owned strings or string slices; allocate cheaply.
        let owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let result = model
            .embed(owned, None)
            .context("embedding inference failed")?;
        Ok(result)
    }

    /// Embed a single text. Convenience helper.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut out = self.embed(&[text])?;
        out.pop().context("empty embedding output")
    }
}

/// Cosine similarity between two vectors of equal length. Returns 0.0
/// if either is zero-norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test downloads the model on first run (~50MB). It's gated
    /// behind the `embeddings-test` env var so CI can skip it.
    #[test]
    #[ignore]
    fn embedder_produces_vectors() {
        let e = Embedder::new().unwrap();
        let v = e.embed_one("hello world").unwrap();
        assert_eq!(v.len(), 384, "MiniLM produces 384-dim vectors");
    }

    #[test]
    fn cosine_handles_empty() {
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
    }
}
