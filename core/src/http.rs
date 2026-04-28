//! HTTP server for Smriti — exposes the memory engine over JSON + SMRP.
//!
//! Routes (JSON):
//! - `POST /api/remember`     — store a memory
//! - `POST /api/recall`       — query memories within a token budget
//! - `POST /api/forget`       — soft-delete a memory
//! - `POST /api/supersede`    — replace one memory with another
//! - `POST /api/link`         — add an edge between two memories
//! - `POST /api/consolidate`  — force consolidation pass
//! - `GET  /api/stats`        — store statistics
//!
//! Routes (SMRP):
//! - `POST /smrp` — accepts a raw SMRP/1.0 request body, returns a wire response.

#![cfg(feature = "http")]

use std::sync::{Arc, Mutex};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

use crate::node::{MemoryEdge, MemoryKind};
use crate::protocol::{SmrpRequest, SmrpResponse, SmrpVerb};
use crate::scope::Scope;
use crate::Smriti;

/// Shared application state. Smriti is wrapped in a `Mutex` because all
/// write paths require `&mut self`.
pub struct HttpState {
    pub smriti: Mutex<Smriti>,
}

pub fn router(state: Arc<HttpState>) -> Router {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/remember", post(remember))
        .route("/api/recall", post(recall))
        .route("/api/forget", post(forget))
        .route("/api/supersede", post(supersede))
        .route("/api/link", post(link))
        .route("/api/consolidate", post(consolidate))
        .route("/api/stats", get(stats))
        .route("/smrp", post(smrp_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Start the HTTP server on the given port.
pub async fn serve(state: Arc<HttpState>, port: u16) -> anyhow::Result<()> {
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    tracing::info!("Smriti HTTP server listening on port {}", port);
    axum::serve(listener, app).await?;
    Ok(())
}

// ── helpers ──

fn ise<E: std::fmt::Display>(e: E) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
}

fn parse_scope(s: &Option<ScopePayload>) -> Scope {
    match s {
        Some(p) => {
            let mut scope = Scope::agent(p.agent_id.as_deref().unwrap_or("default"));
            if let Some(u) = &p.user_id {
                scope.user_id = Some(u.clone());
            }
            if let Some(s) = &p.session_id {
                scope.session_id = Some(s.clone());
            }
            scope
        }
        None => Scope::default(),
    }
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct ScopePayload {
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
}

// ── routes ──

async fn health() -> &'static str {
    "Smriti is alive"
}

#[derive(Deserialize)]
struct RememberReq {
    text: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    importance: Option<f32>,
    #[serde(default)]
    supersedes: Option<Uuid>,
    #[serde(default)]
    scope: Option<ScopePayload>,
}

#[derive(Serialize)]
struct RememberResp {
    id: Uuid,
}

async fn remember(State(state): State<Arc<HttpState>>, Json(req): Json<RememberReq>) -> Response {
    let mut smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };
    let mut builder = smriti
        .remember(req.text)
        .scope(parse_scope(&req.scope))
        .kind(MemoryKind::parse(req.kind.as_deref().unwrap_or("fact")));
    if !req.tags.is_empty() {
        builder = builder.tags(req.tags);
    }
    if let Some(imp) = req.importance {
        builder = builder.importance(imp);
    }
    if let Some(old) = req.supersedes {
        builder = builder.supersedes(old);
    }
    match builder.commit() {
        Ok(id) => Json(RememberResp { id }).into_response(),
        Err(e) => ise(e),
    }
}

#[derive(Deserialize)]
struct RecallReq {
    query: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    budget: Option<usize>,
    #[serde(default)]
    lambda: Option<f32>,
    #[serde(default)]
    kinds: Option<Vec<String>>,
    #[serde(default)]
    scope: Option<ScopePayload>,
}

#[derive(Serialize)]
struct RecallHitResp {
    id: Uuid,
    text: String,
    tags: Vec<String>,
    kind: String,
    final_score: f32,
    fingerprint_sim: f32,
    ppr_score: f32,
    decay_factor: f32,
    from_hippocampus: bool,
}

#[derive(Serialize)]
struct RecallResp {
    hits: Vec<RecallHitResp>,
    tokens_used: usize,
    tokens_budget: usize,
    candidates_considered: usize,
    seeds_used: usize,
}

async fn recall(State(state): State<Arc<HttpState>>, Json(req): Json<RecallReq>) -> Response {
    let smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };

    let mut builder = smriti.recall(req.query).scope(parse_scope(&req.scope));
    if !req.tags.is_empty() {
        builder = builder.tags(req.tags);
    }
    if let Some(b) = req.budget {
        builder = builder.budget(b);
    }
    if let Some(l) = req.lambda {
        builder = builder.lambda(l);
    }
    if let Some(kinds) = req.kinds {
        builder = builder.kinds(kinds.iter().map(|s| MemoryKind::parse(s)).collect());
    }

    match builder.execute() {
        Ok(result) => {
            let resp = RecallResp {
                tokens_used: result.tokens_used,
                tokens_budget: result.tokens_budget,
                candidates_considered: result.candidates_considered,
                seeds_used: result.seeds_used,
                hits: result
                    .hits
                    .into_iter()
                    .map(|h| RecallHitResp {
                        id: h.node.id,
                        text: h.node.text,
                        tags: h.node.tags,
                        kind: h.node.kind.to_string(),
                        final_score: h.final_score,
                        fingerprint_sim: h.fingerprint_sim,
                        ppr_score: h.ppr_score,
                        decay_factor: h.decay_factor,
                        from_hippocampus: h.from_hippocampus,
                    })
                    .collect(),
            };
            Json(resp).into_response()
        }
        Err(e) => ise(e),
    }
}

#[derive(Deserialize)]
struct ForgetReq {
    id: Uuid,
}

async fn forget(State(state): State<Arc<HttpState>>, Json(req): Json<ForgetReq>) -> Response {
    let mut smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };
    match smriti.forget(req.id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => ise(e),
    }
}

#[derive(Deserialize)]
struct SupersedeReq {
    old_id: Uuid,
    new_id: Uuid,
}

async fn supersede(State(state): State<Arc<HttpState>>, Json(req): Json<SupersedeReq>) -> Response {
    let mut smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };
    match smriti.supersede(req.old_id, req.new_id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => ise(e),
    }
}

#[derive(Deserialize)]
struct LinkReq {
    from: Uuid,
    to: Uuid,
    edge: Option<String>,
}

async fn link(State(state): State<Arc<HttpState>>, Json(req): Json<LinkReq>) -> Response {
    let mut smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };
    let edge_str = req.edge.as_deref().unwrap_or("relates_to");
    let edge = MemoryEdge::parse(edge_str).unwrap_or(MemoryEdge::RelatesTo);
    match smriti.link(req.from, req.to, edge) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => ise(e),
    }
}

async fn consolidate(State(state): State<Arc<HttpState>>) -> Response {
    let mut smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };
    match smriti.consolidate() {
        Ok(report) => Json(serde_json::json!({
            "processed": report.processed,
            "promoted": report.promoted,
            "reinforced": report.reinforced,
            "dropped": report.dropped,
            "edges_created": report.edges_created,
        }))
        .into_response(),
        Err(e) => ise(e),
    }
}

async fn stats(State(state): State<Arc<HttpState>>) -> Response {
    let smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };
    match smriti.stats() {
        Ok(s) => Json(s).into_response(),
        Err(e) => ise(e),
    }
}

// ── SMRP handler ──

async fn smrp_handler(State(state): State<Arc<HttpState>>, body: String) -> Response {
    let req = match SmrpRequest::parse(&body) {
        Ok(r) => r,
        Err(e) => {
            return SmrpResponse::error(400, format!("parse error: {}", e))
                .to_wire()
                .into_response();
        }
    };

    let smriti = match state.smriti.lock() {
        Ok(g) => g,
        Err(e) => return ise(format!("lock poisoned: {}", e)),
    };

    match req.verb {
        SmrpVerb::Recall => {
            let scope = req.scope();
            let tags = req.tags();
            let budget = req.header_usize("Budget", 2000);
            let lambda = req.header_f32("Lambda", 0.7);
            let mut b = smriti
                .recall(req.body.clone())
                .scope(scope)
                .budget(budget)
                .lambda(lambda);
            if !tags.is_empty() {
                b = b.tags(tags);
            }
            match b.execute() {
                Ok(result) => {
                    let mut body = String::new();
                    for h in &result.hits {
                        body.push_str(&format!("# {}\n", h.node.id));
                        body.push_str(&format!("score={:.3}\n", h.final_score));
                        body.push_str(&h.node.text);
                        body.push_str("\n\n");
                    }
                    SmrpResponse::ok(body)
                        .header("Tokens-Used", result.tokens_used.to_string())
                        .header("Tokens-Budget", result.tokens_budget.to_string())
                        .header("Hits", result.hits.len().to_string())
                        .header("Candidates", result.candidates_considered.to_string())
                        .to_wire()
                        .into_response()
                }
                Err(e) => SmrpResponse::error(500, e.to_string())
                    .to_wire()
                    .into_response(),
            }
        }
        SmrpVerb::Stats => match smriti.stats() {
            Ok(s) => SmrpResponse::ok(serde_json::to_string(&s).unwrap_or_default())
                .header("Content-Type", "application/json")
                .to_wire()
                .into_response(),
            Err(e) => SmrpResponse::error(500, e.to_string())
                .to_wire()
                .into_response(),
        },
        _ => SmrpResponse::error(501, "verb not implemented over SMRP yet")
            .to_wire()
            .into_response(),
    }
}
