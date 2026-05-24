from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.gateway.models import ConversationSource
from alpha_agent.gateway.session import GatewaySessionStore, SessionMode
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.models import MemoryCandidate, MemoryScope
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.time import utc_now_iso


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def test_debug_prompt_loads_gateway_source_from_session_and_prints_retrieval_trace(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source = ConversationSource(
        platform="feishu",
        chat_id="chat-1",
        chat_type="group",
        user_id="user-1",
        user_name="Ada",
        thread_id="thread-9",
    )
    mapping = GatewaySessionStore(store).get_or_create(source, SessionMode.THREAD_PER_USER)
    episode = EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision",
        source_event_ids=[],
        salience=0.9,
        scope=MemoryScope.from_record(mapping.memory_scope),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "debug",
            "prompt",
            "sqlite memory",
            "--session",
            mapping.session_id,
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert mapping.session_id in result.output
    assert "feishu" in result.output
    assert "chat-1" in result.output
    assert "user-1" in result.output
    assert "thread-9" in result.output
    assert episode.id in result.output
    assert "retrieval_score=" in result.output
    assert "access_count=" in result.output
    assert "access_count=1" not in result.output
    assert store.list_episodic_memories(limit=1)[0].access_count == 0
    with store.connect() as conn:
        access_logs = conn.execute("SELECT count(*) FROM memory_access_log").fetchone()[0]
    assert access_logs == 0


def test_debug_prompt_group_shared_uses_persisted_channel_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = ConversationSource(
        platform="feishu",
        chat_id="shared-chat",
        chat_type="group",
        user_id="user-1",
        user_name="Ada",
    )
    mapping = GatewaySessionStore(store).get_or_create(source, SessionMode.GROUP_SHARED)
    shared_episode = EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision for shared channel",
        source_event_ids=[],
        salience=0.9,
        scope=MemoryScope.from_record(mapping.memory_scope),
    )
    EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision from default user",
        source_event_ids=[],
        salience=0.9,
        scope=MemoryScope.default(),
    )
    EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision from platform user",
        source_event_ids=[],
        salience=0.9,
        scope=MemoryScope(
            kind="platform_user",
            scope_key="platform:feishu:user:user-1",
            platform="feishu",
            user_id="user-1",
        ),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "debug",
            "prompt",
            "sqlite memory",
            "--session",
            mapping.session_id,
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert f"Memory scope: {mapping.memory_scope['scope_key']}" in result.output
    assert shared_episode.id in result.output
    assert "default user" not in result.output
    assert "platform user" not in result.output


def test_memory_search_does_not_record_memory_access(tmp_path: Path) -> None:
    store = _store(tmp_path)
    EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision",
        source_event_ids=[],
        salience=0.9,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["memory", "search", "sqlite memory"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "episodic" in result.output
    assert store.list_episodic_memories(limit=1)[0].access_count == 0
    with store.connect() as conn:
        access_logs = conn.execute("SELECT count(*) FROM memory_access_log").fetchone()[0]
    assert access_logs == 0


def test_skills_list_does_not_load_builtin_skills(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["skills", "list"], env=_env(tmp_path))

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert store.stats()["procedural"] == 0
    assert "Debug Loop" not in result.output


def test_debug_prompt_does_not_load_builtin_skills(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "debug this failing command"],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert store.stats()["procedural"] == 0
    assert "Debug Loop" not in result.output


def test_debug_prompt_manual_source_flags_override_gateway_mapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = ConversationSource(
        platform="feishu",
        chat_id="chat-1",
        chat_type="group",
        user_id="user-1",
        thread_id="thread-9",
    )
    mapping = GatewaySessionStore(store).get_or_create(source, SessionMode.THREAD_PER_USER)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "debug",
            "prompt",
            "hello",
            "--session",
            mapping.session_id,
            "--platform",
            "telegram",
            "--chat-id",
            "chat-override",
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "telegram" in result.output
    assert "chat-override" in result.output
    assert "thread-9" in result.output


def test_memory_review_preview_does_not_store_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["memory", "review", "remember that I prefer tea", "--session", "session-review"],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Candidate" in result.output
    assert "User prefers: tea" in result.output
    assert store.list_semantic_memories() == []
    assert store.list_episodic_memories() == []


def test_memory_review_approve_all_stores_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--approve-all",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Approved 2 candidate" in result.output
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert len(store.list_episodic_memories()) == 1
    assert len(store.list_memory_candidates(status="approved")) == 2
    assert store.stats()["memory_decisions"] == 4


def test_memory_list_filters_default_scope_and_active_semantic_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    semantic = SemanticMemoryManager(store)
    other_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:other",
        platform="telegram",
        user_id="other",
    )
    semantic.upsert_fact("user", "prefers", "tea", "Default active memory")
    semantic.upsert_fact(
        "user",
        "prefers",
        "stale tea",
        "Default inactive memory",
        status="deleted",
    )
    semantic.upsert_fact(
        "user",
        "prefers",
        "coffee",
        "Other scope active memory",
        scope=other_scope,
    )
    EpisodicMemoryManager(store).create(
        "Other scoped episode",
        ["evt-other"],
        salience=0.7,
        scope=other_scope,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "list"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Default active memory" in result.output
    assert "Default inactive memory" not in result.output
    assert "Other scope active memory" not in result.output
    assert "Other scoped episode" not in result.output


def test_memory_forget_marks_semantic_memory_deleted_and_hides_it(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    memory = SemanticMemoryManager(store).upsert_fact(
        "user",
        "prefers",
        "tea",
        "User prefers tea",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "forget", memory.id], env=_env(tmp_path))
    list_result = runner.invoke(app, ["memory", "list"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "deleted" in result.output
    assert memory.id in result.output
    assert list_result.exit_code == 0
    assert "User prefers tea" not in list_result.output
    assert store.list_semantic_memories(statuses=["deleted"])[0].id == memory.id


def test_memory_forget_rejects_out_of_scope_semantic_memory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    other_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:other",
        platform="telegram",
        user_id="other",
    )
    memory = SemanticMemoryManager(store).upsert_fact(
        "user",
        "prefers",
        "coffee",
        "Other scope active memory",
        scope=other_scope,
    )
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "forget", memory.id], env=_env(tmp_path))

    assert result.exit_code != 0
    assert "outside visible scope" in result.output
    assert store.get_semantic_memory(memory.id).status == "active"


def test_memory_audit_shows_supersession_chain_and_sources(tmp_path: Path) -> None:
    store = _store(tmp_path)
    semantic = SemanticMemoryManager(store)
    old = semantic.upsert_fact(
        "user",
        "prefers",
        "tea",
        "User prefers tea",
        source_memory_ids=["msg-old"],
    )
    new = semantic.upsert_fact(
        "user",
        "prefers",
        "coffee",
        "User prefers coffee",
        source_memory_ids=["msg-new"],
    )
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "audit", new.id], env=_env(tmp_path))

    assert result.exit_code == 0
    assert new.id in result.output
    assert old.id in result.output
    assert "msg-new" in result.output
    assert "msg-old" in result.output
    assert "superseded" in result.output


def test_memory_review_reject_all_does_not_store_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--reject-all",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Rejected 2 candidate" in result.output
    assert store.list_semantic_memories() == []
    assert store.list_episodic_memories() == []
    rejected = store.list_memory_candidates(status="rejected")
    assert len(rejected) == 2
    for candidate in rejected:
        assert [decision.action for decision in store.list_memory_decisions(candidate.id)] == [
            "reject"
        ]


def test_memory_review_approve_one_unedited_candidate(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--approve",
            "1",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Approved 1 candidate" in result.output
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert store.list_episodic_memories() == []


def test_memory_review_mixed_approve_and_reject_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--approve",
            "1",
            "--reject",
            "2",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Rejected 1 candidate" in result.output
    assert "Approved 1 candidate" in result.output
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert store.list_episodic_memories() == []
    rejected = store.list_memory_candidates(status="rejected")
    assert len(rejected) == 1
    assert [decision.action for decision in store.list_memory_decisions(rejected[0].id)] == [
        "reject"
    ]


def test_memory_review_edit_one_while_approving_selected_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "I prefer coffee",
            "--session",
            "session-review",
            "--approve",
            "1",
            "--candidate",
            "1",
            "--edit-content",
            "User prefers: tea",
            "--edit-object",
            "tea",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    semantic = store.list_semantic_memories()
    assert result.exit_code == 0
    assert "Edited candidate 1" in result.output
    assert "Approved 1 candidate" in result.output
    assert len(semantic) == 1
    assert semantic[0].content == "User prefers: tea"
    assert semantic[0].object == "tea"


def test_memory_review_lists_and_approves_stored_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-cli-1",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers tea",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=["msg-1"],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    runner = CliRunner()

    listed = runner.invoke(
        app,
        ["memory", "review", "--list-pending"],
        env=_env(tmp_path),
    )
    approved = runner.invoke(
        app,
        [
            "memory",
            "review",
            "--candidate-id",
            "cand-cli-1",
            "--approve-stored",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert listed.exit_code == 0
    assert "cand-cli-1" in listed.output
    assert "scope=user:default" in listed.output
    assert approved.exit_code == 0
    assert "Approved 1 candidate" in approved.output
    assert store.get_memory_candidate("cand-cli-1").status == "approved"
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert store.stats()["memory_decisions"] == 2


def test_memory_review_edits_and_inspects_stored_candidate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = store.append_conversation_message(
        session_id="session-review",
        role="user",
        raw_content="Remember that I prefer coffee",
    )
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-cli-edit",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers coffee",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "coffee"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=[source.id],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    runner = CliRunner()

    edited = runner.invoke(
        app,
        [
            "memory",
            "review",
            "--candidate-id",
            "cand-cli-edit",
            "--edit-stored",
            "--edit-content",
            "User prefers tea",
            "--edit-object",
            "tea",
        ],
        env=_env(tmp_path),
    )
    inspected = runner.invoke(
        app,
        [
            "memory",
            "review",
            "--candidate-id",
            "cand-cli-edit",
            "--inspect-stored",
        ],
        env=_env(tmp_path),
    )
    listed_pending = runner.invoke(
        app,
        ["memory", "review", "--list-pending"],
        env=_env(tmp_path),
    )
    listed_stored = runner.invoke(
        app,
        ["memory", "review", "--list-stored"],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    candidate = store.get_memory_candidate("cand-cli-edit")
    assert edited.exit_code == 0
    assert inspected.exit_code == 0
    assert listed_pending.exit_code == 0
    assert listed_stored.exit_code == 0
    assert "Edited stored candidate cand-cli-edit" in edited.output
    assert "Remember that I prefer coffee" in inspected.output
    assert "action=edit" in inspected.output
    assert "original_content=User prefers coffee" in inspected.output
    assert "edited_content=User prefers tea" in inspected.output
    assert "cand-cli-edit" not in listed_pending.output
    assert "cand-cli-edit" in listed_stored.output
    assert candidate is not None
    assert candidate.status == "edited"
    assert candidate.content == "User prefers tea"
    assert candidate.source_message_ids == [source.id]


def test_memory_review_list_flags_conflict_with_stored_actions(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "--list-pending",
            "--candidate-id",
            "cand-cli-edit",
            "--approve-stored",
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code != 0
    assert "Use only one stored review action" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["unexpected message", "--list-pending"],
        ["--list-stored", "--approve", "1"],
        ["--list-stored", "--reject", "1"],
        ["--list-stored", "--approve-all"],
        ["--list-stored", "--reject-all"],
        ["--list-stored", "--candidate", "1"],
        ["--list-stored", "--edit-content", "User prefers tea"],
        ["--list-stored", "--candidate-id", "cand-1"],
        ["--list-stored", "--session", "session-review"],
        ["unexpected message", "--candidate-id", "cand-1", "--inspect-stored"],
        ["--candidate-id", "cand-1", "--inspect-stored", "--session", "session-review"],
        ["--candidate-id", "cand-1", "--inspect-stored", "--approve", "1"],
        ["--candidate-id", "cand-1", "--inspect-stored", "--reject", "1"],
        ["--candidate-id", "cand-1", "--inspect-stored", "--approve-all"],
        ["--candidate-id", "cand-1", "--inspect-stored", "--reject-all"],
        ["--candidate-id", "cand-1", "--inspect-stored", "--candidate", "1"],
        [
            "--candidate-id",
            "cand-1",
            "--inspect-stored",
            "--edit-content",
            "User prefers tea",
        ],
    ],
)
def test_memory_review_stored_and_list_modes_reject_one_shot_flags(
    tmp_path: Path,
    args: list[str],
) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "review", *args], env=_env(tmp_path))

    assert result.exit_code != 0
    assert "Stored review actions cannot be combined" in result.output


def test_memory_review_list_pending_does_not_cross_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = utc_now_iso()
    other_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:other",
        platform="telegram",
        user_id="other",
    )
    for candidate_id, scope in [
        ("cand-default", MemoryScope.default()),
        ("cand-other", other_scope),
    ]:
        store.insert_memory_candidate(
            MemoryCandidate(
                id=candidate_id,
                candidate_type="semantic",
                proposed_layer="semantic",
                content=f"{candidate_id} content",
                weak_structure={"subject": "user", "predicate": "prefers", "object": candidate_id},
                salience=0.9,
                confidence=0.8,
                scope=scope,
                source_message_ids=["msg-1"],
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "review", "--list-pending"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "cand-default" in result.output
    assert "cand-other" not in result.output


def test_memory_review_stored_candidate_requires_pending_default_scope(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = utc_now_iso()
    other_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:other",
        platform="telegram",
        user_id="other",
    )
    for candidate_id, scope in [
        ("cand-approve", MemoryScope.default()),
        ("cand-reject", MemoryScope.default()),
        ("cand-other", other_scope),
    ]:
        store.insert_memory_candidate(
            MemoryCandidate(
                id=candidate_id,
                candidate_type="semantic",
                proposed_layer="semantic",
                content=f"{candidate_id} content",
                weak_structure={"subject": "user", "predicate": "prefers", "object": candidate_id},
                salience=0.9,
                confidence=0.8,
                scope=scope,
                source_message_ids=["msg-1"],
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
    runner = CliRunner()

    approved = runner.invoke(
        app,
        ["memory", "review", "--candidate-id", "cand-approve", "--approve-stored"],
        env=_env(tmp_path),
    )
    approved_again = runner.invoke(
        app,
        ["memory", "review", "--candidate-id", "cand-approve", "--approve-stored"],
        env=_env(tmp_path),
    )
    reject_approved = runner.invoke(
        app,
        ["memory", "review", "--candidate-id", "cand-approve", "--reject-stored"],
        env=_env(tmp_path),
    )
    rejected = runner.invoke(
        app,
        ["memory", "review", "--candidate-id", "cand-reject", "--reject-stored"],
        env=_env(tmp_path),
    )
    approve_rejected = runner.invoke(
        app,
        ["memory", "review", "--candidate-id", "cand-reject", "--approve-stored"],
        env=_env(tmp_path),
    )
    approve_other_scope = runner.invoke(
        app,
        ["memory", "review", "--candidate-id", "cand-other", "--approve-stored"],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert approved.exit_code == 0
    assert rejected.exit_code == 0
    assert approved_again.exit_code != 0
    assert reject_approved.exit_code != 0
    assert approve_rejected.exit_code != 0
    assert approve_other_scope.exit_code != 0
    assert store.get_memory_candidate("cand-approve").status == "approved"
    assert store.get_memory_candidate("cand-reject").status == "rejected"
    assert store.get_memory_candidate("cand-other").status == "pending"
