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

CREATE TABLE IF NOT EXISTS session_context_states (
    session_id TEXT PRIMARY KEY,
    compressed_until_ordinal INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    summary_source_message_ids TEXT NOT NULL DEFAULT '[]',
    compression_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    CHECK (compressed_until_ordinal >= 0)
);

CREATE TABLE IF NOT EXISTS runtime_traces (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    source_event_ids TEXT NOT NULL DEFAULT '[]',
    people TEXT NOT NULL DEFAULT '[]',
    places TEXT NOT NULL DEFAULT '[]',
    topics TEXT NOT NULL DEFAULT '[]',
    salience REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,
    scope_kind TEXT NOT NULL DEFAULT 'global_user',
    scope_key TEXT NOT NULL DEFAULT 'user:default',
    scope_metadata TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS semantic_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'fact',
    subject TEXT,
    predicate TEXT,
    object TEXT,
    entities TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    salience REAL NOT NULL DEFAULT 0.5,
    stability REAL NOT NULL DEFAULT 0.5,
    source_memory_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    valid_from TEXT,
    valid_until TEXT,
    supersedes_id TEXT,
    superseded_by_id TEXT,
    deleted_at TEXT,
    scope_kind TEXT NOT NULL DEFAULT 'global_user',
    scope_key TEXT NOT NULL DEFAULT 'user:default',
    scope_metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    CHECK (status IN ('active', 'superseded', 'deleted', 'conflict_review'))
);

CREATE TABLE IF NOT EXISTS procedural_memories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    trigger TEXT NOT NULL,
    procedure_markdown TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    scope_kind TEXT NOT NULL DEFAULT 'global_user',
    scope_key TEXT NOT NULL DEFAULT 'user:default',
    scope_metadata TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    UNIQUE(name, scope_key)
);

CREATE TABLE IF NOT EXISTS memory_candidates (
    id TEXT PRIMARY KEY,
    candidate_type TEXT NOT NULL,
    proposed_layer TEXT NOT NULL,
    content TEXT NOT NULL,
    weak_structure TEXT NOT NULL DEFAULT '{}',
    salience REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    scope_kind TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    scope_metadata TEXT NOT NULL DEFAULT '{}',
    source_message_ids TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer_metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memory_decisions (
    id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    action TEXT NOT NULL,
    memory_type TEXT,
    memory_id TEXT,
    reviewer TEXT,
    rationale TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(candidate_id) REFERENCES memory_candidates(id)
);

CREATE TABLE IF NOT EXISTS entity_nodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    kind TEXT,
    aliases TEXT NOT NULL DEFAULT '[]',
    salience REAL NOT NULL DEFAULT 0.5,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS relation_edges (
    id TEXT PRIMARY KEY,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    relation_type TEXT NOT NULL DEFAULT 'related_to',
    evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS memory_access_log (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    query TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    scope_key TEXT NOT NULL DEFAULT 'user:default',
    metadata TEXT NOT NULL DEFAULT '{}'
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
    memory_scope TEXT NOT NULL DEFAULT '{}',
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
CREATE INDEX IF NOT EXISTS idx_session_context_states_updated_at
    ON session_context_states(updated_at);
CREATE INDEX IF NOT EXISTS idx_runtime_traces_session_timestamp
    ON runtime_traces(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_runtime_traces_event_type_timestamp
    ON runtime_traces(event_type, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodic_created_at ON episodic_memories(created_at);
CREATE INDEX IF NOT EXISTS idx_episodic_salience ON episodic_memories(salience);
CREATE INDEX IF NOT EXISTS idx_episodic_scope ON episodic_memories(scope_key);
CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_memories(subject);
CREATE INDEX IF NOT EXISTS idx_semantic_predicate ON semantic_memories(predicate);
CREATE INDEX IF NOT EXISTS idx_semantic_salience ON semantic_memories(salience);
CREATE INDEX IF NOT EXISTS idx_semantic_scope_status ON semantic_memories(scope_key, status);
CREATE INDEX IF NOT EXISTS idx_semantic_scope_structure
    ON semantic_memories(scope_key, subject, predicate, object, status);
CREATE INDEX IF NOT EXISTS idx_semantic_scope_content
    ON semantic_memories(scope_key, normalized_content, status);
CREATE INDEX IF NOT EXISTS idx_procedural_name_scope ON procedural_memories(name, scope_key);
CREATE INDEX IF NOT EXISTS idx_procedural_scope ON procedural_memories(scope_key);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_status
    ON memory_candidates(status, scope_key, updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_decisions_candidate
    ON memory_decisions(candidate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_entity_nodes_salience ON entity_nodes(salience);
CREATE UNIQUE INDEX IF NOT EXISTS idx_relation_edges_unique
    ON relation_edges(source_node_id, target_node_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_memory_access_memory ON memory_access_log(memory_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_gateway_session_lookup
    ON gateway_session_mappings(platform, session_mode, session_key);
CREATE INDEX IF NOT EXISTS idx_gateway_session_session_id
    ON gateway_session_mappings(session_id);
CREATE INDEX IF NOT EXISTS idx_gateway_dedup_platform_message
    ON gateway_dedup(platform, platform_message_id);
CREATE INDEX IF NOT EXISTS idx_gateway_dedup_expires_at
    ON gateway_dedup(expires_at);
