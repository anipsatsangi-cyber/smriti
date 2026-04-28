//! SMRP — Smriti Memory Recall Protocol.
//!
//! A simple, HTTP-style line-oriented wire format for memory operations.
//! Designed to be:
//!
//! - **Human-readable** — debuggable with `nc`, `curl`, browser DevTools.
//! - **Streamable** — request/response are line-delimited; SSE-friendly.
//! - **Versioned** — `SMRP/1.0` lets the protocol evolve.
//! - **Self-describing** — every response carries explicit token accounting.
//!
//! # Request shape
//!
//! ```text
//! SMRP/1.0 RECALL
//! Scope: agent=default; user=alice
//! Budget: 2000
//! Query: how does auth work
//! Tags: auth, security
//! Lambda: 0.7
//! ---
//! ```
//!
//! # Response shape
//!
//! ```text
//! SMRP/1.0 200 OK
//! Tokens-Used: 187
//! Tokens-Budget: 2000
//! Hits: 3
//! Candidates: 47
//! ---
//! [memory blocks separated by blank lines]
//! ```

pub mod smrp;

pub use smrp::{SmrpRequest, SmrpResponse, SmrpVerb};
