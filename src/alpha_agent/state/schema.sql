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
