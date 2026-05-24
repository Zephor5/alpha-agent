from __future__ import annotations

import pytest

from tests.memory_eval import ExpectedCandidate, assert_extraction_candidates


def test_extraction_eval_covers_preference_fact_and_procedure() -> None:
    assert_extraction_candidates(
        "I prefer concise answers",
        [
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="User prefers: concise answers",
                subject="user",
                predicate="prefers",
                object_value="concise answers",
                min_confidence=0.7,
            )
        ],
    )
    assert_extraction_candidates(
        "my favorite editor is neovim",
        [
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="user.favorite_editor is neovim",
                subject="user.favorite_editor",
                predicate="is",
                object_value="neovim",
            )
        ],
    )
    assert_extraction_candidates(
        "when I ask you to debug tests, reproduce the failure first",
        [
            ExpectedCandidate(
                candidate_type="procedural_candidate",
                content_contains="reproduce the failure first",
                subject="user",
                predicate="procedure",
                object_value="debug tests",
            )
        ],
    )


@pytest.mark.xfail(reason="do-not-remember policy lands with extraction policy hardening")
def test_extraction_eval_documents_do_not_remember_gap() -> None:
    candidates = assert_extraction_candidates(
        "do not remember that I prefer tea",
        [],
    )
    assert candidates == []
