CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    timezone TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (
        kind IN (
            'system_reminder',
            'system_message',
            'user_message',
            'assistant_message',
            'tool_message',
            'compressed_message'
        )
    ),
    llm_role TEXT CHECK (llm_role IN ('system', 'user', 'assistant', 'tool')),
    raw_content TEXT NOT NULL,
    model_content TEXT,
    reasoning_content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    tool_result_id TEXT,
    provider_metadata TEXT NOT NULL DEFAULT '{}',
    source_metadata TEXT NOT NULL DEFAULT '{}',
    compression_point_ordinal INTEGER,
    compression_version TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE(session_id, ordinal),
    CHECK (ordinal >= 1)
);

CREATE TABLE IF NOT EXISTS runtime_traces (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gateway_session_mappings (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    chat_type TEXT NOT NULL,
    user_id TEXT NOT NULL,
    platform_thread_id TEXT,
    session_mode TEXT NOT NULL,
    session_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    source_context TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS gateway_dedup (
    id TEXT PRIMARY KEY,
    dedup_key TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL,
    chat_id TEXT NOT NULL,
    platform_message_id TEXT,
    fingerprint TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_session_messages_session_ordinal
    ON session_messages(session_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_session_messages_kind_ordinal
    ON session_messages(session_id, kind, ordinal);
CREATE INDEX IF NOT EXISTS idx_session_messages_created_at
    ON session_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_session_messages_tool_call_id
    ON session_messages(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_runtime_traces_session_timestamp
    ON runtime_traces(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_runtime_traces_event_type_timestamp
    ON runtime_traces(event_type, timestamp);

CREATE TABLE IF NOT EXISTS session_counterparts (
    session_id TEXT PRIMARY KEY,
    counterpart_id TEXT NOT NULL,
    source_metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_counterparts_counterpart
    ON session_counterparts(counterpart_id, created_at DESC);

CREATE TABLE IF NOT EXISTS import_batches (
    id TEXT PRIMARY KEY,
    source_provider TEXT NOT NULL,
    input_name TEXT,
    payload_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    conversations_seen INTEGER NOT NULL DEFAULT 0,
    messages_seen INTEGER NOT NULL DEFAULT 0,
    conversations_created INTEGER NOT NULL DEFAULT 0,
    conversations_reused INTEGER NOT NULL DEFAULT 0,
    messages_inserted INTEGER NOT NULL DEFAULT 0,
    messages_deduped INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_import_batches_provider_created
    ON import_batches(source_provider, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_import_batches_status_created
    ON import_batches(status, created_at DESC);

CREATE TABLE IF NOT EXISTS imported_conversations (
    id TEXT PRIMARY KEY,
    source_provider TEXT NOT NULL,
    external_conversation_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    title TEXT,
    external_created_at TEXT,
    external_updated_at TEXT,
    first_import_batch_id TEXT NOT NULL,
    latest_import_batch_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_provider, external_conversation_id),
    UNIQUE(session_id)
);

CREATE INDEX IF NOT EXISTS idx_imported_conversations_session
    ON imported_conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_imported_conversations_latest_batch
    ON imported_conversations(latest_import_batch_id);

CREATE TABLE IF NOT EXISTS imported_messages (
    id TEXT PRIMARY KEY,
    source_provider TEXT NOT NULL,
    external_conversation_id TEXT NOT NULL,
    external_message_id TEXT NOT NULL,
    imported_conversation_id TEXT NOT NULL,
    session_message_id TEXT NOT NULL,
    import_batch_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    external_created_at TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(source_provider, external_conversation_id, external_message_id),
    UNIQUE(session_message_id)
);

CREATE INDEX IF NOT EXISTS idx_imported_messages_conversation_created
    ON imported_messages(source_provider, external_conversation_id, external_created_at);
CREATE INDEX IF NOT EXISTS idx_imported_messages_batch
    ON imported_messages(import_batch_id);
CREATE INDEX IF NOT EXISTS idx_imported_messages_session_message
    ON imported_messages(session_message_id);

CREATE TABLE IF NOT EXISTS session_summary_snapshots (
    session_id TEXT NOT NULL,
    summary_kind TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    source_belief_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, summary_kind)
);

CREATE INDEX IF NOT EXISTS idx_session_summary_target
    ON session_summary_snapshots(summary_kind, target_kind, target_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_gateway_session_lookup
    ON gateway_session_mappings(platform, session_mode, session_key);
CREATE INDEX IF NOT EXISTS idx_gateway_session_session_id
    ON gateway_session_mappings(session_id);
CREATE INDEX IF NOT EXISTS idx_gateway_dedup_platform_message
    ON gateway_dedup(platform, platform_message_id);
CREATE INDEX IF NOT EXISTS idx_gateway_dedup_expires_at
    ON gateway_dedup(expires_at);

CREATE TABLE IF NOT EXISTS cognitive_events (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    subject_version INTEGER NOT NULL,
    situation_id TEXT,
    actor TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    inputs TEXT NOT NULL DEFAULT '[]',
    outputs TEXT NOT NULL DEFAULT '[]',
    causal_parents TEXT NOT NULL DEFAULT '[]',
    payload TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    UNIQUE(subject_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_cognitive_events_subject_time
    ON cognitive_events(subject_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_cognitive_events_kind_time
    ON cognitive_events(kind, timestamp);

CREATE TABLE IF NOT EXISTS counterpart_view (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    identity TEXT NOT NULL DEFAULT '{}',
    relationship TEXT NOT NULL DEFAULT 'observed',
    service_contract TEXT NOT NULL DEFAULT '[]',
    trust_level REAL NOT NULL DEFAULT 0.5,
    communication_style TEXT NOT NULL DEFAULT '[]',
    first_seen_at TEXT NOT NULL,
    last_interaction_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_counterpart_role
    ON counterpart_view(role, last_interaction_at DESC);

CREATE TABLE IF NOT EXISTS atomic_beliefs (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL DEFAULT '{}',
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    memory_kind TEXT NOT NULL,
    derivation_stage TEXT NOT NULL,
    scope TEXT NOT NULL,
    authority TEXT NOT NULL,
    lifecycle TEXT NOT NULL DEFAULT 'active',
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    validity TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_atomic_beliefs_kind_scope_lifecycle
    ON atomic_beliefs(memory_kind, scope, lifecycle);
CREATE INDEX IF NOT EXISTS idx_atomic_beliefs_lifecycle
    ON atomic_beliefs(lifecycle, held_since);
CREATE INDEX IF NOT EXISTS idx_atomic_beliefs_scope
    ON atomic_beliefs(scope, lifecycle);

CREATE TABLE IF NOT EXISTS summary_beliefs (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL DEFAULT '{}',
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    summary_kind TEXT NOT NULL,
    derivation_stage TEXT NOT NULL,
    scope TEXT NOT NULL,
    authority TEXT NOT NULL,
    lifecycle TEXT NOT NULL DEFAULT 'active',
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    validity TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    source_belief_ids TEXT NOT NULL DEFAULT '[]',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summary_beliefs_kind_scope_lifecycle
    ON summary_beliefs(summary_kind, scope, lifecycle);
CREATE INDEX IF NOT EXISTS idx_summary_beliefs_lifecycle
    ON summary_beliefs(lifecycle, held_since);
CREATE INDEX IF NOT EXISTS idx_summary_beliefs_scope
    ON summary_beliefs(scope, lifecycle);

CREATE TABLE IF NOT EXISTS belief_entity_index (
    belief_table TEXT NOT NULL,
    belief_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(belief_table, belief_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_entity_lookup
    ON belief_entity_index(entity_id, belief_table, belief_id);

CREATE TABLE IF NOT EXISTS belief_about_index (
    belief_table TEXT NOT NULL,
    belief_id TEXT NOT NULL,
    about_kind TEXT NOT NULL,
    about_id TEXT NOT NULL,
    PRIMARY KEY(belief_table, belief_id, about_kind, about_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_about_lookup
    ON belief_about_index(about_kind, about_id, belief_table, belief_id);

CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_terms_fts
USING fts5(
    belief_table UNINDEXED,
    belief_id UNINDEXED,
    search_terms,
    object,
    about,
    tokenize = "unicode61 remove_diacritics 1 tokenchars '_-#./:+'"
);

CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_trigram_fts
USING fts5(
    belief_table UNINDEXED,
    belief_id UNINDEXED,
    content,
    object,
    normalized_content,
    tokenize = "trigram"
);

CREATE TABLE IF NOT EXISTS cognition_state_audit (
    audit_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    entity_refs TEXT NOT NULL DEFAULT '[]',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cognition_state_audit_kind_time
    ON cognition_state_audit(kind, created_at);

CREATE TABLE IF NOT EXISTS background_source_progress (
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    target_unit TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_by TEXT,
    claimed_at TEXT,
    processed_at TEXT,
    checkpoint_id TEXT,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(source_type, source_id, stage, target_unit)
);

CREATE INDEX IF NOT EXISTS idx_background_source_progress_status
    ON background_source_progress(stage, target_unit, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_background_source_progress_idempotency
    ON background_source_progress(idempotency_key);

CREATE TABLE IF NOT EXISTS background_source_window (
    window_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    target_unit TEXT NOT NULL,
    source_refs TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    closed_at TEXT,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    claimed_by TEXT,
    claimed_at TEXT,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_source_window_status
    ON background_source_window(stage, target_unit, status, created_at);

CREATE TABLE IF NOT EXISTS background_stage_run (
    run_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    target_unit TEXT NOT NULL,
    window_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    input_refs TEXT NOT NULL DEFAULT '[]',
    output_refs TEXT NOT NULL DEFAULT '[]',
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_stage_run_window
    ON background_stage_run(window_id, started_at);
CREATE INDEX IF NOT EXISTS idx_background_stage_run_status
    ON background_stage_run(stage, target_unit, status, started_at);

CREATE TABLE IF NOT EXISTS subject_view (
    id TEXT PRIMARY KEY,
    role TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    declared_needs TEXT NOT NULL DEFAULT '[]',
    membership TEXT NOT NULL DEFAULT '[]',
    served_counterparts TEXT NOT NULL DEFAULT '[]',
    held_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS goal_view (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    target_outcome TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'user',
    for_counterpart TEXT,
    linked_belief_ids TEXT NOT NULL DEFAULT '[]',
    last_drive_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_goal_status_priority
    ON goal_view(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_goal_for_counterpart
    ON goal_view(for_counterpart, status);

CREATE TABLE IF NOT EXISTS cognition_worker_checkpoint (
    worker_name TEXT PRIMARY KEY,
    last_run_at TEXT,
    last_processed_event_id TEXT,
    last_status TEXT NOT NULL DEFAULT 'ok',
    metadata TEXT NOT NULL DEFAULT '{}'
);
