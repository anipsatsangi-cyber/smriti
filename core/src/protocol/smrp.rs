//! SMRP/1.0 parser & serializer.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::scope::Scope;

/// SMRP verbs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmrpVerb {
    Recall,
    Remember,
    Forget,
    Supersede,
    Link,
    Snapshot,
    Stats,
}

impl SmrpVerb {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "RECALL" => Some(SmrpVerb::Recall),
            "REMEMBER" => Some(SmrpVerb::Remember),
            "FORGET" => Some(SmrpVerb::Forget),
            "SUPERSEDE" => Some(SmrpVerb::Supersede),
            "LINK" => Some(SmrpVerb::Link),
            "SNAPSHOT" => Some(SmrpVerb::Snapshot),
            "STATS" => Some(SmrpVerb::Stats),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SmrpVerb::Recall => "RECALL",
            SmrpVerb::Remember => "REMEMBER",
            SmrpVerb::Forget => "FORGET",
            SmrpVerb::Supersede => "SUPERSEDE",
            SmrpVerb::Link => "LINK",
            SmrpVerb::Snapshot => "SNAPSHOT",
            SmrpVerb::Stats => "STATS",
        }
    }
}

/// A parsed SMRP request.
#[derive(Debug, Clone)]
pub struct SmrpRequest {
    pub version: String,
    pub verb: SmrpVerb,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl SmrpRequest {
    /// Parse an SMRP request from a textual blob.
    pub fn parse(input: &str) -> Result<Self> {
        let mut lines = input.lines();
        let first = lines
            .next()
            .ok_or_else(|| anyhow!("empty SMRP request"))?
            .trim();

        let mut parts = first.splitn(2, char::is_whitespace);
        let version = parts
            .next()
            .ok_or_else(|| anyhow!("missing SMRP version"))?
            .to_string();
        let verb_str = parts
            .next()
            .ok_or_else(|| anyhow!("missing SMRP verb"))?
            .trim();

        if !version.starts_with("SMRP/") {
            return Err(anyhow!("not an SMRP request: '{}'", version));
        }

        let verb = SmrpVerb::parse(verb_str)
            .ok_or_else(|| anyhow!("unknown SMRP verb: '{}'", verb_str))?;

        let mut headers = HashMap::new();
        let mut body_lines: Vec<&str> = Vec::new();
        let mut in_body = false;

        for line in lines {
            if !in_body {
                let trimmed = line.trim_end();
                if trimmed == "---" {
                    in_body = true;
                    continue;
                }
                if trimmed.is_empty() {
                    continue;
                }
                if let Some((k, v)) = trimmed.split_once(':') {
                    headers.insert(k.trim().to_string(), v.trim().to_string());
                }
            } else {
                body_lines.push(line);
            }
        }

        Ok(Self {
            version,
            verb,
            headers,
            body: body_lines.join("\n"),
        })
    }

    /// Convenience: parse the `Scope` header.
    pub fn scope(&self) -> Scope {
        let raw = self.headers.get("Scope").cloned().unwrap_or_default();
        let mut scope = Scope::default();
        for part in raw.split(';') {
            let part = part.trim();
            if let Some(v) = part.strip_prefix("agent=") {
                scope.agent_id = v.to_string();
            } else if let Some(v) = part.strip_prefix("user=") {
                scope.user_id = Some(v.to_string());
            } else if let Some(v) = part.strip_prefix("session=") {
                scope.session_id = Some(v.to_string());
            }
        }
        scope
    }

    /// Convenience: parse a numeric header with a default.
    pub fn header_usize(&self, name: &str, default: usize) -> usize {
        self.headers
            .get(name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Convenience: parse a float header with a default.
    pub fn header_f32(&self, name: &str, default: f32) -> f32 {
        self.headers
            .get(name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Convenience: get a comma-separated tag list.
    pub fn tags(&self) -> Vec<String> {
        self.headers
            .get("Tags")
            .map(|s| {
                s.split(',')
                    .map(|t| t.trim().to_string())
                    .filter(|t| !t.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// An SMRP response. Status codes follow HTTP conventions.
#[derive(Debug, Clone)]
pub struct SmrpResponse {
    pub version: String,
    pub status: u16,
    pub status_text: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl SmrpResponse {
    pub fn ok(body: impl Into<String>) -> Self {
        Self {
            version: "SMRP/1.0".to_string(),
            status: 200,
            status_text: "OK".to_string(),
            headers: HashMap::new(),
            body: body.into(),
        }
    }

    pub fn error(status: u16, msg: impl Into<String>) -> Self {
        Self {
            version: "SMRP/1.0".to_string(),
            status,
            status_text: msg.into(),
            headers: HashMap::new(),
            body: String::new(),
        }
    }

    pub fn header(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.headers.insert(key.into(), val.into());
        self
    }

    /// Render to wire format.
    pub fn to_wire(&self) -> String {
        let mut out = format!("{} {} {}\n", self.version, self.status, self.status_text);
        let mut keys: Vec<&String> = self.headers.keys().collect();
        keys.sort();
        for k in keys {
            out.push_str(k);
            out.push_str(": ");
            out.push_str(&self.headers[k]);
            out.push('\n');
        }
        out.push_str("---\n");
        out.push_str(&self.body);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recall_request() {
        let raw = "SMRP/1.0 RECALL\n\
                   Scope: agent=default; user=alice\n\
                   Budget: 1500\n\
                   Tags: auth, security\n\
                   ---\n\
                   how does authentication work";
        let req = SmrpRequest::parse(raw).unwrap();
        assert_eq!(req.verb, SmrpVerb::Recall);
        assert_eq!(req.header_usize("Budget", 0), 1500);
        let scope = req.scope();
        assert_eq!(scope.agent_id, "default");
        assert_eq!(scope.user_id.as_deref(), Some("alice"));
        assert!(req.body.contains("authentication"));
        let tags = req.tags();
        assert_eq!(tags, vec!["auth", "security"]);
    }

    #[test]
    fn unknown_verb_rejected() {
        let raw = "SMRP/1.0 EXPLODE\n---\n";
        assert!(SmrpRequest::parse(raw).is_err());
    }

    #[test]
    fn response_to_wire_is_deterministic() {
        let resp = SmrpResponse::ok("body content")
            .header("Tokens-Used", "187")
            .header("Hits", "3");
        let wire = resp.to_wire();
        assert!(wire.starts_with("SMRP/1.0 200 OK\n"));
        assert!(wire.contains("Hits: 3"));
        assert!(wire.contains("Tokens-Used: 187"));
        assert!(wire.contains("---\nbody content"));
    }

    #[test]
    fn missing_scope_returns_default() {
        let raw = "SMRP/1.0 RECALL\nBudget: 100\n---\nhi";
        let req = SmrpRequest::parse(raw).unwrap();
        assert_eq!(req.scope(), Scope::default());
    }
}
