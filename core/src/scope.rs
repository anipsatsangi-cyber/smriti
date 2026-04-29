//! Multi-tenant scope for memory partitioning.
//!
//! Every memory belongs to exactly one `Scope`. Recall is **scope-cascading**:
//! a query at session-level transparently includes user-level and
//! agent-level memories, but never sees memories from a different user or
//! agent.
//!
//! ```text
//! agent="default"
//!   ↓
//! user="alice"      ← inherits agent
//!   ↓
//! session="s_42"    ← inherits user, inherits agent
//! ```

use serde::{Deserialize, Serialize};

/// Multi-tenant scope. `agent_id` is required; `user_id` and `session_id`
/// are optional and provide finer partitioning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope {
    /// Required. The agent or namespace this memory belongs to.
    pub agent_id: String,
    /// Optional. The user this memory is about / belongs to.
    pub user_id: Option<String>,
    /// Optional. The session this memory was created in. Sessions are
    /// short-lived; session-scoped memories typically don't survive
    /// consolidation into the neocortex.
    pub session_id: Option<String>,
    /// Optional. Other agents that are explicitly allowed to read this memory.
    #[serde(default)]
    pub shared_with: Vec<String>,
}

impl Scope {
    /// Create a new scope with just an agent_id.
    pub fn agent(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            user_id: None,
            session_id: None,
            shared_with: Vec::new(),
        }
    }

    /// Add a user_id to the scope (builder).
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Add a session_id to the scope (builder).
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Share this scope with another agent (builder).
    pub fn share_with(mut self, agent_id: impl Into<String>) -> Self {
        let agent_id = agent_id.into();
        if !self.shared_with.contains(&agent_id) {
            self.shared_with.push(agent_id);
        }
        self
    }

    /// Whether this scope can read memories from `other`.
    ///
    /// Cascading rules:
    /// - Different agents → no access
    /// - Same agent, no user → reads all agent-level memories
    /// - Same agent + user → reads agent-level AND that user's memories
    /// - Same agent + user + session → reads all of the above plus that session
    pub fn can_read(&self, other: &Scope) -> bool {
        if self.agent_id != other.agent_id {
            if other.shared_with.contains(&self.agent_id) {
                return true;
            }
            return false;
        }
        // Agent-level memories (no user) are visible to everyone in the agent
        if other.user_id.is_none() {
            return true;
        }
        // User-level memories require matching user
        if self.user_id != other.user_id {
            return false;
        }
        // Session-level memories require matching session OR no session on self
        if let Some(other_session) = &other.session_id {
            if let Some(my_session) = &self.session_id {
                return my_session == other_session;
            }
            // Memory is session-scoped but reader has no session → cannot read
            return false;
        }
        true
    }

    /// Compact string representation for SQLite storage.
    pub fn to_key(&self) -> String {
        let user = self.user_id.as_deref().unwrap_or("");
        let session = self.session_id.as_deref().unwrap_or("");
        if self.shared_with.is_empty() {
            format!("{}|{}|{}", self.agent_id, user, session)
        } else {
            format!("{}|{}|{}|{}", self.agent_id, user, session, self.shared_with.join(","))
        }
    }

    /// Parse a Scope back from its compact key form.
    pub fn from_key(key: &str) -> Option<Self> {
        let parts: Vec<&str> = key.splitn(4, '|').collect();
        if parts.len() < 3 {
            return None;
        }
        
        let shared_with = if parts.len() == 4 && !parts[3].is_empty() {
            parts[3].split(',').map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        };

        Some(Self {
            agent_id: parts[0].to_string(),
            user_id: if parts[1].is_empty() {
                None
            } else {
                Some(parts[1].to_string())
            },
            session_id: if parts[2].is_empty() {
                None
            } else {
                Some(parts[2].to_string())
            },
            shared_with,
        })
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::agent("default")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_roundtrip() {
        let s = Scope::agent("default")
            .with_user("alice")
            .with_session("s_42");
        let key = s.to_key();
        let s2 = Scope::from_key(&key).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn agent_only_scope_roundtrips() {
        let s = Scope::agent("acme");
        let s2 = Scope::from_key(&s.to_key()).unwrap();
        assert_eq!(s, s2);
    }

    #[test]
    fn cascading_read_rules() {
        let agent_only = Scope::agent("a");
        let alice_agent = Scope::agent("a").with_user("alice");
        let bob_agent = Scope::agent("a").with_user("bob");
        let alice_session = Scope::agent("a").with_user("alice").with_session("s1");

        // Same agent, no user: visible to everyone
        assert!(alice_agent.can_read(&agent_only));
        assert!(bob_agent.can_read(&agent_only));

        // Different users in same agent: cannot read each other
        assert!(!alice_agent.can_read(&bob_agent));

        // Session can read its user's memories
        assert!(alice_session.can_read(&alice_agent));

        // User cannot peek into a specific session
        assert!(!alice_agent.can_read(&alice_session));
    }

    #[test]
    fn different_agents_isolated() {
        let a = Scope::agent("agent_a");
        let b = Scope::agent("agent_b");
        assert!(!a.can_read(&b));
        assert!(!b.can_read(&a));
    }

    #[test]
    fn federated_sharing() {
        let a = Scope::agent("agent_a");
        let b = Scope::agent("agent_b").share_with("agent_a");
        let c = Scope::agent("agent_c");
        
        // a can read b because b shared it with a
        assert!(a.can_read(&b));
        // b cannot read a
        assert!(!b.can_read(&a));
        // c cannot read b
        assert!(!c.can_read(&b));
    }

    #[test]
    fn shared_key_roundtrip() {
        let s = Scope::agent("default")
            .share_with("agent_b")
            .share_with("agent_c");
        let key = s.to_key();
        let s2 = Scope::from_key(&key).unwrap();
        assert_eq!(s, s2);
        assert_eq!(s2.shared_with, vec!["agent_b", "agent_c"]);
    }
}
