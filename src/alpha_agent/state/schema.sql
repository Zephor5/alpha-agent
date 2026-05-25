CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    raw_content TEXT NOT NULL,
    model_content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT NOT NULL DEFAULT '[]',
    tool_result_id TEXT,
    provider_metadata TEXT NOT NULL DEFAULT '{}',
    source_metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
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
    thread_id TEXT,
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

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_ordinal
    ON conversation_messages(session_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_created_at
    ON conversation_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_tool_call_id
    ON conversation_messages(tool_call_id);
CREATE INDEX IF NOT EXISTS idx_runtime_traces_session_timestamp
    ON runtime_traces(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_runtime_traces_event_type_timestamp
    ON runtime_traces(event_type, timestamp);
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

CREATE TABLE IF NOT EXISTS belief_view (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL DEFAULT '{}',
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    cognitive_type TEXT NOT NULL,
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    applicability TEXT NOT NULL DEFAULT '{}',
    value_profile TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_belief_view_status
    ON belief_view(status);
CREATE INDEX IF NOT EXISTS idx_belief_view_type
    ON belief_view(cognitive_type, status);

CREATE TABLE IF NOT EXISTS belief_entity_index (
    belief_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(belief_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_entity_lookup
    ON belief_entity_index(entity_id, belief_id);

CREATE TABLE IF NOT EXISTS belief_about_index (
    belief_id TEXT NOT NULL,
    about_kind TEXT NOT NULL,
    about_id TEXT NOT NULL,
    PRIMARY KEY(belief_id, about_kind, about_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_about_lookup
    ON belief_about_index(about_kind, about_id, belief_id);

CREATE TABLE IF NOT EXISTS context_window_view (
    thread_id TEXT PRIMARY KEY,
    thread_kind TEXT NOT NULL,
    counterpart_id TEXT,
    foreground_ids TEXT NOT NULL DEFAULT '[]',
    anchored_ids TEXT NOT NULL DEFAULT '[]',
    recent_judgment_ids TEXT NOT NULL DEFAULT '[]',
    matched_procedure_ids TEXT NOT NULL DEFAULT '[]',
    background_summary_id TEXT,
    last_event_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ctx_window_counterpart
    ON context_window_view(counterpart_id, thread_kind);
CREATE INDEX IF NOT EXISTS idx_ctx_window_kind
    ON context_window_view(thread_kind);

CREATE TABLE IF NOT EXISTS context_window_background (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    derived_from_perception_ids TEXT NOT NULL DEFAULT '[]',
    preserved_anchors TEXT NOT NULL DEFAULT '[]',
    compression_policy TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ctx_bg_thread_time
    ON context_window_background(thread_id, created_at DESC);

CREATE TABLE IF NOT EXISTS reflection_view (
    id TEXT PRIMARY KEY,
    tick_id TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'L1',
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    finding TEXT NOT NULL,
    suggested_remedy TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflection_severity
    ON reflection_view(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflection_kind
    ON reflection_view(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS strategy_view (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    target_stages TEXT NOT NULL DEFAULT '[]',
    for_counterpart TEXT,
    set_by TEXT NOT NULL,
    set_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_status_validity
    ON strategy_view(status, valid_until);
CREATE INDEX IF NOT EXISTS idx_strategy_for_counterpart
    ON strategy_view(for_counterpart, status);

CREATE TABLE IF NOT EXISTS subject_view (
    id TEXT PRIMARY KEY,
    role TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    declared_needs TEXT NOT NULL DEFAULT '[]',
    value_lens_id TEXT,
    self_model TEXT NOT NULL DEFAULT '{}',
    served_counterparts TEXT NOT NULL DEFAULT '[]',
    known_biases TEXT NOT NULL DEFAULT '[]',
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

CREATE TABLE IF NOT EXISTS subject_value_lens (
    subject_id TEXT PRIMARY KEY,
    priority TEXT NOT NULL,
    sensitivity TEXT NOT NULL DEFAULT '{}',
    tradeoff_preferences TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS procedure_view (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    steps TEXT NOT NULL DEFAULT '[]',
    expected_outcome TEXT NOT NULL DEFAULT '',
    learned_from_event_ids TEXT NOT NULL DEFAULT '[]',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_procedure_trigger
    ON procedure_view(trigger_pattern);

CREATE TABLE IF NOT EXISTS cognition_worker_checkpoint (
    worker_name TEXT PRIMARY KEY,
    last_run_at TEXT,
    last_processed_event_id TEXT,
    last_status TEXT NOT NULL DEFAULT 'ok',
    metadata TEXT NOT NULL DEFAULT '{}'
);
