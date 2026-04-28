//! Hyperdimensional Computing (HDC) — Binary Spatter Codes.
//!
//! This module implements **Binary Spatter Codes** (BSC), one of the
//! Vector Symbolic Architectures introduced by Pentti Kanerva (1996) and
//! formalized in Plate (1995, "Holographic Reduced Representations").
//!
//! # Why HDC for memory?
//!
//! Traditional memory systems store text and search by similarity. HDC
//! lets us treat memories as **algebraic objects** that can be composed
//! and decomposed without lookup tables.
//!
//! Three operations on `Hypervector`s:
//!
//! - **Bind** (`⊗`, XOR) — associates two concepts into one. Reversible.
//!   `key ⊗ value = composite`, and `composite ⊗ key = value`.
//! - **Bundle** (`+`, majority vote) — superposes multiple vectors. Lossy
//!   but recoverable: each component is still recognizable in the bundle.
//! - **Permute** (`Π`) — rotates bits. Used for sequence encoding.
//!
//! # Capacity
//!
//! With dimension `D = 2048`, we can store ~`D / log₂(D)` ≈ 186 bindings
//! in a single bundle and still recover each one with > 99% accuracy
//! (Kanerva 2009, capacity bound).
//!
//! Two random hypervectors are nearly orthogonal: their Hamming distance
//! is ≈ D/2 with standard deviation `√D/2`. This means random hypervectors
//! are distinguishable at high confidence: P(collision) ≈ 2^(-D/8) for
//! D=2048 = 2^(-256), astronomically small.
//!
//! # Why binary?
//!
//! Binary ops (XOR, popcount, AND, OR) are first-class CPU instructions and
//! WASM instructions. A `bind` of two 2048-bit vectors is 32 XOR ops on
//! u64 lanes. A similarity computation is 32 popcounts. Both run in
//! ~10ns on modern hardware. No FP, no SIMD library required.

use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Dimensionality of hypervectors. 2048 bits = 256 bytes = 32 u64 lanes.
///
/// This is the sweet spot: large enough for ~186-binding capacity, small
/// enough that 10K hypervectors fit in 2.5 MB.
pub const HV_DIM: usize = 2048;

/// Number of u64 lanes per hypervector (2048 / 64).
pub const HV_LANES: usize = HV_DIM / 64;

/// A binary hypervector. Each of the 2048 bits is 0 or 1.
///
/// Stored as 32 u64 lanes for fast XOR/popcount.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Hypervector {
    pub bits: [u64; HV_LANES],
}

impl Hypervector {
    /// The all-zero hypervector. Used as the additive identity.
    pub const fn zero() -> Self {
        Self {
            bits: [0u64; HV_LANES],
        }
    }

    /// Generate a random hypervector deterministically from a seed.
    ///
    /// Uses a simple SplitMix64 PRNG so we can reproduce vectors from a
    /// string seed (same string → same vector). This is critical for
    /// bind/unbind to work across processes and persistence boundaries.
    pub fn from_seed(seed: u64) -> Self {
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut bits = [0u64; HV_LANES];
        for lane in bits.iter_mut() {
            // SplitMix64 — Sebastiano Vigna's mixer
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            *lane = z;
        }
        Self { bits }
    }

    /// Generate a hypervector from a string token. Used for binding.
    pub fn from_token(s: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        Self::from_seed(hasher.finish())
    }

    /// Hamming distance — count of differing bits.
    ///
    /// For two random hypervectors this is ≈ HV_DIM/2 ± √(HV_DIM)/2.
    /// For identical hypervectors it is 0.
    pub fn hamming(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        for i in 0..HV_LANES {
            d += (self.bits[i] ^ other.bits[i]).count_ones();
        }
        d
    }

    /// Cosine-like similarity in [-1, 1].
    ///
    /// Computed as `1 - 2 * hamming / D`. Identical vectors → 1.0.
    /// Orthogonal random vectors → ~0.0. Inverted vectors → -1.0.
    pub fn similarity(&self, other: &Self) -> f32 {
        let h = self.hamming(other) as f32;
        1.0 - 2.0 * h / (HV_DIM as f32)
    }

    /// Number of set bits (popcount).
    pub fn popcount(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Sparsity: fraction of bits that are 1.
    pub fn density(&self) -> f32 {
        self.popcount() as f32 / (HV_DIM as f32)
    }
}

impl PartialEq for Hypervector {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for Hypervector {}

impl std::fmt::Debug for Hypervector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hypervector(density={:.3}, lane0={:016x})",
            self.density(),
            self.bits[0]
        )
    }
}

/// **Bind** — associates two hypervectors via XOR.
///
/// Bind is its own inverse: `bind(bind(a, b), b) == a`. This lets us
/// store key-value pairs as a single vector and recover the value when
/// we know the key.
///
/// Distributive over bundle: `bind(bundle(a, b), c) ≈ bundle(bind(a, c), bind(b, c))`.
pub fn bind(a: &Hypervector, b: &Hypervector) -> Hypervector {
    let mut out = Hypervector::zero();
    for i in 0..HV_LANES {
        out.bits[i] = a.bits[i] ^ b.bits[i];
    }
    out
}

/// **Unbind** — algebraic inverse of bind. For BSC, unbind == bind (XOR
/// is self-inverse). Provided as a named operation for clarity.
pub fn unbind(composite: &Hypervector, key: &Hypervector) -> Hypervector {
    bind(composite, key)
}

/// **Bundle** — superposes multiple hypervectors via majority vote.
///
/// Each output bit is 1 if more than half of inputs have that bit set.
/// Ties (when N is even) are broken by a deterministic tie-breaker
/// hypervector. The result is similar to all inputs simultaneously,
/// so we can ask "is X in this bundle?" by checking similarity.
///
/// Capacity: a 2048-bit BSC bundle holds ~186 bindings recoverable at
/// >99% accuracy (Kanerva 2009).
pub fn bundle(vectors: &[&Hypervector]) -> Hypervector {
    if vectors.is_empty() {
        return Hypervector::zero();
    }
    if vectors.len() == 1 {
        return *vectors[0];
    }

    let n = vectors.len();
    let threshold = n / 2;

    // For each bit position, count set bits across input vectors.
    // To avoid an O(D) array of u32 counts, we process 64 bits at a time
    // by accumulating per-lane bit counts.
    let mut out = Hypervector::zero();

    for lane in 0..HV_LANES {
        // Count, for each of the 64 bits in this lane, how many vectors
        // have that bit set.
        let mut counts = [0u16; 64];
        for v in vectors {
            let word = v.bits[lane];
            for (b, count) in counts.iter_mut().enumerate() {
                if word & (1u64 << b) != 0 {
                    *count += 1;
                }
            }
        }
        // Majority vote per bit.
        let mut result = 0u64;
        for (b, &count) in counts.iter().enumerate() {
            if count as usize > threshold {
                result |= 1u64 << b;
            } else if count as usize == threshold && n % 2 == 0 {
                // Tie-breaker: use a deterministic vector based on bit
                // position. Symmetric tie-breaking is critical to
                // preserve algebraic properties.
                let tie = (lane * 64 + b) as u64;
                if (tie.wrapping_mul(0x9E3779B97F4A7C15) & 1) == 1 {
                    result |= 1u64 << b;
                }
            }
        }
        out.bits[lane] = result;
    }

    out
}

/// **Permute** — cyclic right-rotation by 1 bit. Used to encode sequence
/// position: `bind(perm(a), b)` differs from `bind(a, perm(b))` so order
/// matters when needed.
pub fn permute(v: &Hypervector) -> Hypervector {
    let mut out = Hypervector::zero();
    // Rotate the entire 2048-bit value right by 1. The bit shifted out
    // of lane[i] goes into the high bit of lane[i-1] (with wraparound).
    let mut carry = (v.bits[0] & 1) << 63;
    for lane in (0..HV_LANES).rev() {
        let new_carry = (v.bits[lane] & 1) << 63;
        out.bits[lane] = (v.bits[lane] >> 1) | carry;
        carry = new_carry;
    }
    out
}

/// Compose a memory's text + tags into a single hypervector representing
/// "this memory."
///
/// Strategy (Weighted Bundling — Kanerva's repeated superposition):
/// 1. Tags:     3× vote weight — highest signal, user-supplied intent.
/// 2. Unigrams: 2× vote weight — content words carry core semantics.
/// 3. Bigrams:  2× vote weight — `bind(permute(w1), w2)` for sequence.
/// 4. Trigrams:  1× vote weight — morphological robustness, typo tolerance.
///
/// By repeating high-signal components in the bundle, they win more
/// majority-vote bits without changing the algebraic properties.
/// Without weighting, ~58 trigrams would drown ~15 word-level components
/// in a typical sentence, compressing similarity to [0.08, 0.19].
pub fn fingerprint(text: &str, tags: &[String]) -> Hypervector {
    let mut components: Vec<Hypervector> = Vec::with_capacity(tags.len() * 3 + 128);

    // 1. Tags — 3× vote weight
    for tag in tags {
        let tv = Hypervector::from_token(&format!("tag::{}", tag));
        components.push(tv);
        components.push(tv);
        components.push(tv);
    }

    let stopwords = [
        "the", "and", "a", "an", "is", "in", "it", "of", "to", "for", "with", "on", "at", "by",
        "this", "that", "are", "we", "our", "you",
    ];

    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| w.len() >= 2)
        .collect();

    // 2. Unigrams (2×) & Bigrams (2×)
    let mut prev_vec: Option<Hypervector> = None;
    for word in &words {
        let is_stop = stopwords.contains(&word.as_str());
        let w_vec = Hypervector::from_token(&format!("word::{}", word));

        if !is_stop {
            // Unigram — 2× vote weight
            components.push(w_vec);
            components.push(w_vec);
        }

        // Bigram — 2× vote weight (include stopwords for syntax context)
        if let Some(p_vec) = prev_vec {
            let bigram = bind(&permute(&p_vec), &w_vec);
            components.push(bigram);
            components.push(bigram);
        }
        prev_vec = Some(w_vec);
    }

    // 3. Character Trigrams — 1× vote weight (morphological background)
    let raw_lower = text.to_lowercase();
    let chars: Vec<char> = raw_lower.chars().collect();
    if chars.len() >= 3 {
        for window in chars.windows(3) {
            let tri: String = window.iter().collect();
            components.push(Hypervector::from_token(&format!("tri::{}", tri)));
        }
    } else if !chars.is_empty() {
        components.push(Hypervector::from_token(&raw_lower));
    }

    if components.is_empty() {
        return Hypervector::from_token(text);
    }

    let refs: Vec<&Hypervector> = components.iter().collect();
    bundle(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_vectors_are_orthogonal() {
        let a = Hypervector::from_seed(1);
        let b = Hypervector::from_seed(2);
        let sim = a.similarity(&b);
        // Two independent random vectors should have similarity near 0
        // with stddev ~1/sqrt(D) ≈ 0.022 for D=2048.
        assert!(sim.abs() < 0.15, "similarity = {}", sim);
    }

    #[test]
    fn identical_vectors_have_similarity_one() {
        let a = Hypervector::from_seed(42);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bind_is_self_inverse() {
        let key = Hypervector::from_token("user_name");
        let value = Hypervector::from_token("Bob");
        let composite = bind(&key, &value);

        // Unbind with key recovers value
        let recovered = unbind(&composite, &key);
        assert_eq!(recovered, value);

        // Also: composite is far from both inputs (orthogonal-ish)
        assert!(composite.similarity(&value).abs() < 0.15);
        assert!(composite.similarity(&key).abs() < 0.15);
    }

    #[test]
    fn bundle_preserves_membership() {
        // Encode multiple bindings into a single bundle, then verify each
        // is recoverable.
        let name_key = Hypervector::from_token("name");
        let role_key = Hypervector::from_token("role");
        let pref_key = Hypervector::from_token("pref");

        let bob = Hypervector::from_token("Bob");
        let engineer = Hypervector::from_token("engineer");
        let concise = Hypervector::from_token("concise");

        let user_profile = bundle(&[
            &bind(&name_key, &bob),
            &bind(&role_key, &engineer),
            &bind(&pref_key, &concise),
        ]);

        // Query: what is the name? → unbind(profile, name_key) should
        // be much closer to "Bob" than to "engineer" or "concise"
        let query = unbind(&user_profile, &name_key);
        let sim_bob = query.similarity(&bob);
        let sim_eng = query.similarity(&engineer);
        let sim_con = query.similarity(&concise);

        assert!(
            sim_bob > sim_eng && sim_bob > sim_con,
            "expected Bob to win: sim_bob={}, sim_eng={}, sim_con={}",
            sim_bob,
            sim_eng,
            sim_con,
        );
        // Confidence should be reasonably high
        assert!(sim_bob > 0.2, "sim_bob = {}", sim_bob);
    }

    #[test]
    fn permute_changes_vector() {
        let v = Hypervector::from_seed(7);
        let p = permute(&v);
        assert_ne!(v, p);
        assert!(v.similarity(&p).abs() < 0.15);
    }

    #[test]
    fn fingerprint_similar_for_similar_text() {
        let f1 = fingerprint(
            "The auth module uses JWT RS256",
            &["auth".to_string(), "security".to_string()],
        );
        let f2 = fingerprint(
            "Authentication uses JWT with RS256 algorithm",
            &["auth".to_string(), "security".to_string()],
        );
        let f3 = fingerprint(
            "User prefers concise responses",
            &["preference".to_string()],
        );

        let close = f1.similarity(&f2);
        let far = f1.similarity(&f3);
        assert!(
            close > far,
            "similar texts should be closer: close={}, far={}",
            close,
            far
        );
    }

    #[test]
    fn fingerprint_token_seeded() {
        // Same input → same output (determinism)
        let a = fingerprint("hello world", &["test".to_string()]);
        let b = fingerprint("hello world", &["test".to_string()]);
        assert_eq!(a, b);
    }
}
