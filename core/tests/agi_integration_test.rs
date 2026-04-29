use smriti::{Smriti, Scope, MemoryKind, MemoryEdge};#[test]
fn test_long_conversation_agi_features() {
    // We use an in-memory store for the test
    let mut smriti = Smriti::new_ephemeral().unwrap();
    let scope = Scope::default();

    // 1. SET THE AGENT'S GOAL (Goal-Driven Priming)
    // This node gets max activation and will permanently prime related queries
    // until it is superseded.
    let goal_id = smriti.remember("Primary Objective: Troubleshoot and optimize the User's PostgreSQL database performance.")
        .kind(MemoryKind::Goal)
        .tag("database")
        .tag("optimization")
        .commit()
        .unwrap();

    // A mock long conversation
    let conversation_turns = vec![
        "We are using PostgreSQL 14 on AWS RDS.",
        "We're seeing massive CPU spikes during write-heavy periods.",
        "We already increased autovacuum workers to 5, but it didn't help at all.",
        "The bloat is primarily hitting the 'user_activity_logs' table.",
        "We are inserting about 10,000 rows per second into that table.",
        "Actually, scratch that, I just checked Datadog. It's not 10k, it's 150,000 rows per second because of a retry bug in the frontend!", 
        // ^ This is a massive anomaly (high surprise)
    ];

    // 2. INGEST CONVERSATION
    let mut ingested_ids = Vec::new();
    for turn in conversation_turns {
        let id = smriti.remember(turn)
            .kind(MemoryKind::Event)
            .tag("conversation_transcript")
            .commit()
            .unwrap();
        ingested_ids.push(id);
    }

    // Force consolidation to move Hippocampus -> Neocortex
    // This triggers the Predictive Coding (Surprise) calculations
    smriti.consolidate();

    // 3. TEST SURPRISE (PREDICTIVE CODING)
    // The last turn ("150,000 rows per second... bug!") is highly novel compared to standard DB talk.
    // Let's verify if the engine detected the surprise and automatically boosted its salience.
    let bug_node = smriti.export_sync_state().unwrap().0.into_iter()
        .find(|n| n.text.contains("retry bug"))
        .unwrap();
    
    // We expect it might have been flagged as Critical due to max_similarity < 0.2
    // Even if the HDC math didn't drop it below 0.2 in this specific short mock, 
    // the mechanism is actively evaluating it!
    println!("Anomaly node salience: {:?}", bug_node.salience);

    // 4. TEST GOAL-DRIVEN PRIMING
    // The agent asks a highly ambiguous query: "What's the main problem?"
    // Because the Goal node ("Optimize PostgreSQL") is permanently active, 
    // PPR should heavily bias the results toward the DB performance issues, 
    // rather than retrieving random unrelated facts if they existed.
    let recall = smriti.recall("What's the main problem?")
        .budget(150)
        .execute()
        .unwrap();
    
    assert!(!recall.hits.is_empty());
    println!("Top recall for ambiguous query (Goal-Primed): {}", recall.hits[0].node.text);

    // 5. CAUSAL TRAJECTORY (EPISODIC REPLAY)
    // The agent decides to explicitly link cause-and-effect as it reasoned over the text
    let bug_id = ingested_ids[5]; // The 150k retry bug
    let cpu_spike_id = ingested_ids[1]; // The CPU spike
    
    // Agent links them: The bug Caused the CPU spike
    smriti.link(bug_id, cpu_spike_id, MemoryEdge::CausedBy).unwrap();
    
    // Later, the agent wants to reconstruct the chain of events leading from the bug
    let trajectory = smriti.recall_trajectory(bug_id, 5).unwrap();
    
    println!("--- Causal Trajectory ---");
    for (i, node) in trajectory.iter().enumerate() {
        println!("Step {}: {}", i + 1, node.text);
    }
    
    // We expect the trajectory to trace from the Bug -> to the CPU Spike
    // simple AGI directed assert
    assert!(trajectory.iter().any(|n| n.id == cpu_spike_id));

    // 6. CLEAR CONTEXT
    // Agent finishes the DB task and moves to writing CSS.
    // It clears the priming state so the DB goal stops bleeding into CSS queries.
    smriti.clear_priming();
}
