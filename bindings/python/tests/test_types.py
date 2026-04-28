"""Unit tests for Smriti type models."""

from smriti.types import Memory, RecallHit, RecallResult, Stats, StoreStats


def test_memory_defaults():
    m = Memory(id="abc", text="hello")
    assert m.kind == "fact"
    assert m.importance == 0.5
    assert m.tags == []
    assert m.access_count == 0


def test_memory_full():
    m = Memory(
        id="abc-123",
        text="The auth module uses JWT",
        tags=["auth", "security"],
        kind="decision",
        importance=0.9,
        access_count=5,
        token_count=8,
    )
    assert m.kind == "decision"
    assert len(m.tags) == 2
    assert m.token_count == 8


def test_recall_result_empty():
    r = RecallResult()
    assert r.hits == []
    assert r.tokens_used == 0


def test_recall_result_with_hits():
    hit = RecallHit(
        node=Memory(id="x", text="test memory"),
        final_score=0.85,
        fingerprint_sim=0.7,
        ppr_score=0.15,
    )
    r = RecallResult(hits=[hit], tokens_used=10, tokens_budget=2000)
    assert len(r.hits) == 1
    assert r.hits[0].final_score == 0.85
    assert r.hits[0].node.text == "test memory"


def test_stats():
    s = Stats(
        store=StoreStats(total_memories=100, active_memories=95),
        hippocampus_size=20,
        hippocampus_capacity=128,
        neocortex_size=75,
    )
    assert s.store.total_memories == 100
    assert s.hippocampus_capacity == 128
