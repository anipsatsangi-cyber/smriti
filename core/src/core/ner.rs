//! Lightweight Named Entity Recognition for auto-tagging.
//!
//! This is **not** a real ML-based NER. It's a deterministic rule-based
//! extractor designed to surface candidate tags from memory text without
//! requiring any model. It runs in microseconds and adds no runtime
//! dependencies.
//!
//! What it extracts:
//!
//! 1. **Capitalized phrases** (1–3 consecutive capitalized words) — proper
//!    nouns, project names, product names. "Project Phoenix", "Bob",
//!    "JWT", "ACME Corp".
//! 2. **Domain keywords** — known technical terms (auth, deploy, refactor,
//!    etc.) that frequently appear as tags. These are matched
//!    case-insensitively.
//! 3. **Verb stems** — the action being performed. Not full lemmatization,
//!    just stripping common suffixes (-ing, -ed, -s).
//!
//! The extracted tags are merged with user-supplied tags. Duplicates are
//! filtered. The merged set is what auto-linking uses to find related
//! memories.

use std::collections::HashSet;

/// Common English stopwords excluded from extraction.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "and", "or", "but",
    "if", "then", "so", "this", "that", "these", "those", "have", "has", "had", "do", "does",
    "did", "will", "would", "should", "could", "can", "may", "might", "must", "shall", "what",
    "when", "where", "why", "how", "who", "which", "into", "about", "over", "after", "before",
    "during",
];

/// Domain keywords that often serve as good tags. We match these
/// case-insensitively in the input text.
const DOMAIN_KEYWORDS: &[&str] = &[
    "auth",
    "authentication",
    "authorization",
    "session",
    "token",
    "jwt",
    "oauth",
    "saml",
    "sso",
    "password",
    "login",
    "logout",
    "api",
    "rest",
    "graphql",
    "rpc",
    "grpc",
    "endpoint",
    "webhook",
    "database",
    "db",
    "sql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "cache",
    "queue",
    "kafka",
    "rabbitmq",
    "deploy",
    "release",
    "rollout",
    "rollback",
    "ci",
    "cd",
    "test",
    "testing",
    "qa",
    "bug",
    "issue",
    "ticket",
    "frontend",
    "backend",
    "fullstack",
    "ui",
    "ux",
    "design",
    "performance",
    "latency",
    "throughput",
    "scaling",
    "scale",
    "security",
    "encryption",
    "tls",
    "ssl",
    "vulnerability",
    "audit",
    "monitoring",
    "logging",
    "metrics",
    "tracing",
    "observability",
    "kubernetes",
    "k8s",
    "docker",
    "container",
    "pod",
    "rust",
    "python",
    "typescript",
    "javascript",
    "java",
    "golang",
    "go",
    "refactor",
    "refactoring",
    "optimization",
    "rewrite",
    "user",
    "team",
    "lead",
    "engineer",
    "developer",
    "project",
    "milestone",
    "deadline",
    "sprint",
    "config",
    "configuration",
    "env",
    "environment",
    "memory",
    "recall",
    "remember",
    "forget",
    "supersede",
];

/// Extract candidate tags from a piece of text.
///
/// Returns a deduplicated `Vec<String>` of lowercase tags suitable for
/// merging with user-supplied tags.
pub fn extract_tags(text: &str) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();

    // 1. Capitalized phrases (proper-noun heuristic).
    extract_proper_nouns(text, &mut out);

    // 2. Domain keywords.
    extract_domain_keywords(text, &mut out);

    // 3. Verb stems.
    extract_verb_stems(text, &mut out);

    // Stable order (alphabetical) for testability.
    let mut v: Vec<String> = out.into_iter().collect();
    v.sort();
    v
}

/// Merge extracted tags with user-supplied tags. User tags win on
/// case-conflict; extracted-only tags are added in lowercase form.
pub fn merge_tags(user_tags: &[String], extracted: &[String]) -> Vec<String> {
    let mut out: Vec<String> = user_tags.to_vec();
    let lower_existing: HashSet<String> = out.iter().map(|t| t.to_lowercase()).collect();
    for tag in extracted {
        if !lower_existing.contains(tag) {
            out.push(tag.clone());
        }
    }
    out
}

/// Same as [`merge_tags`], but also returns provenance for each tag.
/// User-supplied tags map to `TagSource::User`; auto-extracted tags map
/// to `TagSource::Auto`. The returned vectors have identical length and
/// indices align — `(tags[i], sources[i])` is the provenance pair.
pub fn merge_tags_with_sources(
    user_tags: &[String],
    extracted: &[String],
) -> (Vec<String>, Vec<crate::node::TagSource>) {
    use crate::node::TagSource;
    let mut tags: Vec<String> = user_tags.to_vec();
    let mut sources: Vec<TagSource> = vec![TagSource::User; tags.len()];
    let lower_existing: HashSet<String> = tags.iter().map(|t| t.to_lowercase()).collect();
    for tag in extracted {
        if !lower_existing.contains(tag) {
            tags.push(tag.clone());
            sources.push(TagSource::Auto);
        }
    }
    (tags, sources)
}

fn extract_proper_nouns(text: &str, out: &mut HashSet<String>) {
    // Split on whitespace AND keep simple punctuation handling.
    let tokens: Vec<&str> = text
        .split(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | ':'))
        .filter(|s| !s.is_empty())
        .collect();

    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        // First letter must be uppercase, and the token must be > 1 char.
        let first_char = tok.chars().next();
        if let Some(c) = first_char {
            if c.is_ascii_uppercase() && tok.len() >= 2 {
                // First-position capitalization at the start of a sentence
                // is too weak a signal — skip if this is the first token
                // overall. Keep mid-sentence capitalized words.
                if i == 0 {
                    // Still extract: the first token may be a project name.
                    // Be conservative — only extract if the second token
                    // is also capitalized (suggesting a multi-word name)
                    // or if this token contains internal capitalization
                    // (suggesting a brand: "GitHub", "MongoDB").
                    let has_internal_caps = tok.chars().skip(1).any(|c| c.is_ascii_uppercase());
                    let next_capped = tokens
                        .get(i + 1)
                        .and_then(|t| t.chars().next())
                        .map(|c| c.is_ascii_uppercase())
                        .unwrap_or(false);
                    if !has_internal_caps && !next_capped {
                        i += 1;
                        continue;
                    }
                }
                let cleaned: String = tok.chars().filter(|c| c.is_alphanumeric()).collect();
                if cleaned.len() >= 2 && !STOPWORDS.contains(&cleaned.to_lowercase().as_str()) {
                    out.insert(cleaned.to_lowercase());
                }
            }
        }
        i += 1;
    }
}

fn extract_domain_keywords(text: &str, out: &mut HashSet<String>) {
    let lower = text.to_lowercase();
    for &kw in DOMAIN_KEYWORDS {
        // Word-boundary check: surrounded by non-alphanumeric.
        let mut from = 0;
        while let Some(pos) = lower[from..].find(kw) {
            let abs = from + pos;
            let before_ok = abs == 0
                || !lower[..abs]
                    .chars()
                    .last()
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false);
            let after_idx = abs + kw.len();
            let after_ok = after_idx >= lower.len()
                || !lower[after_idx..]
                    .chars()
                    .next()
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false);
            if before_ok && after_ok {
                out.insert(kw.to_string());
            }
            from = abs + kw.len();
            if from >= lower.len() {
                break;
            }
        }
    }
}

fn extract_verb_stems(text: &str, out: &mut HashSet<String>) {
    // Heuristic: any word ending in -ing or -ed of length ≥ 5 is treated
    // as a verb. We strip the suffix to get the stem.
    for raw in text.split_whitespace() {
        let lower: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .collect::<String>()
            .to_lowercase();
        if lower.len() < 5 {
            continue;
        }
        let stem: Option<&str> = if lower.ends_with("ing") {
            Some(&lower[..lower.len() - 3])
        } else if lower.ends_with("ed") {
            Some(&lower[..lower.len() - 2])
        } else {
            None
        };
        if let Some(stem) = stem {
            // Drop double-letter endings ("running" → "runn"  → "run")
            let cleaned = if stem.len() >= 4
                && stem.as_bytes()[stem.len() - 1] == stem.as_bytes()[stem.len() - 2]
                && !is_vowel(stem.chars().nth(stem.len() - 1).unwrap())
            {
                &stem[..stem.len() - 1]
            } else {
                stem
            };
            if cleaned.len() >= 3 && !STOPWORDS.contains(&cleaned) {
                out.insert(cleaned.to_string());
            }
        }
    }
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_proper_nouns() {
        let tags = extract_tags("Project Phoenix uses Rust and Tokio");
        assert!(tags.contains(&"phoenix".to_string()));
        assert!(tags.contains(&"rust".to_string()));
    }

    #[test]
    fn extracts_domain_keywords() {
        let tags = extract_tags("The auth module uses JWT for authorization");
        assert!(tags.contains(&"auth".to_string()));
        assert!(tags.contains(&"jwt".to_string()));
        assert!(tags.contains(&"authorization".to_string()));
    }

    #[test]
    fn extracts_verb_stems() {
        let tags = extract_tags("Alice is leading the auth refactoring effort");
        // "leading" → "lead" or "leadin"; "refactoring" → "refactor"
        // We accept either as long as at least one shows the verb.
        assert!(
            tags.iter()
                .any(|t| t.starts_with("lead") || t.starts_with("refact")),
            "expected verb stem in {:?}",
            tags
        );
    }

    #[test]
    fn ignores_stopwords() {
        let tags = extract_tags("The user has been the engineer");
        assert!(!tags.contains(&"the".to_string()));
        assert!(!tags.contains(&"has".to_string()));
        assert!(!tags.contains(&"been".to_string()));
    }

    #[test]
    fn merge_dedupes_case_insensitive() {
        let user = vec!["Auth".to_string(), "Security".to_string()];
        let extracted = vec!["auth".to_string(), "jwt".to_string()];
        let merged = merge_tags(&user, &extracted);
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&"Auth".to_string()));
        assert!(merged.contains(&"Security".to_string()));
        assert!(merged.contains(&"jwt".to_string()));
    }

    #[test]
    fn handles_empty_text() {
        let tags = extract_tags("");
        assert!(tags.is_empty());
    }
}
