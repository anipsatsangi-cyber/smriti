//! Compile-check + runtime sanity check for the "Smriti without an LLM"
//! example used in README and docs/capabilities.md. If a public API
//! drifts, this test fails and the docs get fixed before the next release.

use smriti::{
    AttrFilter, AttributeValue, MemoryEdge, MemoryKind, RecallVerdict, Smriti,
};

#[test]
fn readme_smriti_without_an_llm_example_compiles_and_runs() {
    let mut s = Smriti::open(":memory:").unwrap();

    // 1. Ingest some structured memories with attributes.
    let bug = s
        .remember("retry storm spiked DB writes to 150k rows/sec")
        .kind(MemoryKind::Event)
        .tag("incident")
        .tag("database")
        .attr("severity", AttributeValue::Number(9.0))
        .attr("region", AttributeValue::Text("us-west-2".into()))
        .commit()
        .unwrap();

    let cpu = s
        .remember("CPU saturated to 98% on the primary Postgres node")
        .kind(MemoryKind::Event)
        .tag("incident")
        .tag("database")
        .attr("severity", AttributeValue::Number(8.0))
        .commit()
        .unwrap();

    // 2. Encode causal structure.
    s.link(bug, cpu, MemoryEdge::CausedBy).unwrap();
    s.consolidate().unwrap();

    // 3. Recall with a confidence verdict.
    let r = s
        .recall("what caused the database overload")
        .budget(500)
        .where_attr(
            "severity",
            AttrFilter::Gt(AttributeValue::Number(7.0)),
        )
        .confident_truncation(2, 2, 0)
        .execute()
        .unwrap();

    // The verdict must be one of the expected variants — we don't assert
    // a specific one because tiny corpora can land in several depending
    // on RRF tie-breaks; the point is the surface compiles and produces
    // a structured verdict.
    let _v = match r.verdict {
        RecallVerdict::Confident
        | RecallVerdict::AmbiguousLeader
        | RecallVerdict::UnsupportedTop
        | RecallVerdict::LowConfidence
        | RecallVerdict::Abstained => true,
    };
    assert!(_v);

    // 4. Reconstruct the narrative chain.
    let chain = s.recall_trajectory(bug, 5).unwrap();
    assert!(chain.iter().any(|n| n.id == bug || n.id == cpu));

    // 5. Correct an outdated fact (supersede chain).
    let _corrected = s
        .remember("retry storm peaked at 200k rows/sec, not 150k")
        .kind(MemoryKind::Event)
        .tag("incident")
        .supersedes(bug)
        .commit()
        .unwrap();

    // 6. Topic switch.
    s.clear_activation();
}
