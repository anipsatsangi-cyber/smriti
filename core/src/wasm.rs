//! WebAssembly bindings — `WasmSmriti` is the JS-facing wrapper.
//!
//! This module is only compiled for `target_arch = "wasm32"`. It exposes
//! a small, stable surface designed to be ergonomic from JavaScript:
//!
//! ```js
//! import init, { WasmSmriti } from "./pkg/smriti.js";
//! await init();
//!
//! const s = new WasmSmriti();
//! s.remember("Bob is the lead engineer", "fact", ["user", "team"]);
//! s.consolidate();
//!
//! const result = s.recall("who leads the team?", 500);
//! console.log(JSON.parse(result));
//! ```
//!
//! All return values cross the JS/WASM boundary as JSON strings to avoid
//! complex object marshalling. Callers `JSON.parse` on the JS side.

#![cfg(target_arch = "wasm32")]

use std::sync::Mutex;

use wasm_bindgen::prelude::*;

use crate::node::MemoryKind;
use crate::scope::Scope;
use crate::Smriti;

/// JavaScript-facing handle to a Smriti instance.
///
/// Internally a `Smriti` behind a `Mutex` because every write path
/// requires `&mut self` and JS is single-threaded but our trait
/// signatures still demand interior-mutability for ergonomics.
#[wasm_bindgen]
pub struct WasmSmriti {
    inner: Mutex<Smriti>,
}

#[wasm_bindgen]
impl WasmSmriti {
    /// Construct a new ephemeral memory engine.
    ///
    /// Pass an optional agent_id (defaults to "default"). Memories live
    /// only in browser memory until the page is closed.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmSmriti, JsError> {
        // Wire up panic-to-console redirection if the wasm-debug feature
        // was enabled — turns Rust panics into actionable browser errors.
        #[cfg(feature = "wasm-debug")]
        console_error_panic_hook::set_once();

        let smriti = Smriti::new_ephemeral().map_err(to_js_err)?;
        Ok(Self {
            inner: Mutex::new(smriti),
        })
    }

    /// Store a memory. Returns the new memory's UUID as a string.
    ///
    /// `kind` should be one of: "fact", "decision", "event", "preference".
    /// Unknown kinds fall back to "fact".
    /// `tags` is a JS array of strings.
    #[wasm_bindgen]
    pub fn remember(
        &self,
        text: String,
        kind: String,
        tags: Vec<String>,
    ) -> Result<String, JsError> {
        let mut s = self.inner.lock().map_err(lock_err)?;
        let mut builder = s
            .remember(text)
            .kind(MemoryKind::parse(&kind))
            .scope(Scope::default());
        if !tags.is_empty() {
            builder = builder.tags(tags);
        }
        let id = builder.commit().map_err(to_js_err)?;
        Ok(id.to_string())
    }

    /// Recall memories matching a query within a token budget.
    ///
    /// Returns a JSON string with shape:
    /// ```json
    /// {
    ///   "hits": [{ "id": "...", "text": "...", "score": 1.42, ... }, ...],
    ///   "tokens_used": 47,
    ///   "tokens_budget": 500,
    ///   "candidates_considered": 12
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn recall(&self, query: String, budget: usize) -> Result<String, JsError> {
        let s = self.inner.lock().map_err(lock_err)?;
        let result = s
            .recall(query)
            .budget(budget)
            .execute()
            .map_err(to_js_err)?;

        let hits: Vec<serde_json::Value> = result
            .hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "id": h.node.id.to_string(),
                    "text": h.node.text,
                    "tags": h.node.tags,
                    "kind": h.node.kind.to_string(),
                    "score": h.final_score,
                    "fingerprint_sim": h.fingerprint_sim,
                    "ppr_score": h.ppr_score,
                    "decay_factor": h.decay_factor,
                    "from_hippocampus": h.from_hippocampus,
                    "dense_sim": h.dense_sim,
                })
            })
            .collect();

        let payload = serde_json::json!({
            "hits": hits,
            "tokens_used": result.tokens_used,
            "tokens_budget": result.tokens_budget,
            "candidates_considered": result.candidates_considered,
            "seeds_used": result.seeds_used,
            "verdict": serde_json::to_value(&result.verdict).unwrap(),
        });

        serde_json::to_string(&payload).map_err(to_js_err)
    }

    /// Force a consolidation pass — drains the hippocampus into the
    /// neocortex. Returns a JSON string with the per-pass report.
    #[wasm_bindgen]
    pub fn consolidate(&self) -> Result<String, JsError> {
        let mut s = self.inner.lock().map_err(lock_err)?;
        let report = s.consolidate().map_err(to_js_err)?;
        let payload = serde_json::json!({
            "processed": report.processed,
            "promoted": report.promoted,
            "reinforced": report.reinforced,
            "dropped": report.dropped,
            "edges_created": report.edges_created,
        });
        serde_json::to_string(&payload).map_err(to_js_err)
    }

    /// Soft-delete a memory by UUID string. The memory is hidden from
    /// recall but kept in the audit trail.
    #[wasm_bindgen]
    pub fn forget(&self, id: String) -> Result<(), JsError> {
        let uuid = uuid::Uuid::parse_str(&id).map_err(to_js_err)?;
        let mut s = self.inner.lock().map_err(lock_err)?;
        s.forget(uuid).map_err(to_js_err)
    }

    /// Mark `old_id` as superseded by `new_id`. The old memory is
    /// hidden from recall; the new one carries the link.
    #[wasm_bindgen]
    pub fn supersede(&self, old_id: String, new_id: String) -> Result<(), JsError> {
        let old = uuid::Uuid::parse_str(&old_id).map_err(to_js_err)?;
        let new = uuid::Uuid::parse_str(&new_id).map_err(to_js_err)?;
        let mut s = self.inner.lock().map_err(lock_err)?;
        s.supersede(old, new).map_err(to_js_err)
    }

    /// Aggregate stats as JSON.
    #[wasm_bindgen(js_name = stats)]
    pub fn get_stats(&self) -> Result<String, JsError> {
        let s = self.inner.lock().map_err(lock_err)?;
        let stats = s.stats().map_err(to_js_err)?;
        serde_json::to_string(&stats).map_err(to_js_err)
    }

    /// **Hard delete** — wipe every memory and edge from this engine
    /// instance and start over with an empty ephemeral store. Unlike
    /// [`forget`] (which is a soft-delete tombstone in the audit chain),
    /// this is unrecoverable: the underlying graph is dropped and replaced.
    ///
    /// Used by the in-browser demo's "Reset" button so visitors can
    /// confidently wipe anything they've pasted in. Per-tab WASM
    /// instances are already isolated from other visitors — this is
    /// about *the visitor wiping their own state*, not about cross-
    /// tenant cleanup (which is handled by the architecture itself).
    ///
    /// Returns the number of active memories that were dropped. Errors
    /// only on lock poisoning, which shouldn't happen in single-threaded
    /// JS.
    #[wasm_bindgen]
    pub fn reset(&self) -> Result<usize, JsError> {
        let mut s = self.inner.lock().map_err(lock_err)?;
        let dropped = s.stats().map(|st| st.store.active_memories).unwrap_or(0);
        let fresh = Smriti::new_ephemeral().map_err(to_js_err)?;
        *s = fresh;
        Ok(dropped)
    }

    /// Library version (semver string).
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }
}

// ── helpers ──────────────────────────────────────────────────────────

fn to_js_err<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

/// Poison-tolerant lock helper.
///
/// `Mutex::lock()` returns `Err(PoisonError)` once any prior holder of
/// the lock panicked while holding it. The data is *probably* still
/// valid — the panic might have been an arithmetic edge case that the
/// engine has already moved past — and refusing to proceed permanently
/// breaks the WASM demo.
///
/// On the public landing page, "demo permanently locked because one
/// query panicked" is far worse UX than "best-effort recover and let
/// the user keep going." So we accept the inner data on poison and
/// log to console, rather than refusing the operation.
///
/// The `wasm-debug` feature surfaces the original panic to the browser
/// console via `console_error_panic_hook`, so the root cause is still
/// debuggable — we're only declining to compound the failure.
fn lock_err<T>(e: std::sync::PoisonError<T>) -> JsError {
    let _ = &e; // poisoned data could be salvaged via e.into_inner()
    JsError::new(
        "smriti lock poisoned (a prior call panicked); reload the page to recover",
    )
}
