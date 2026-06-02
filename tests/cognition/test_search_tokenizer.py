from __future__ import annotations

from pathlib import Path

from alpha_agent.cognition.search_tokenizer import cut_jieba_text, tokenize_mixed_text


def test_mixed_tokenizer_preserves_technical_runs_and_derives_parts() -> None:
    tokens = tokenize_mixed_text(
        "v3.0.1 GPT-5.4-mini memory_recall OpenAI API "
        "src/alpha_agent/runtime/agent.py C++ C#"
    )

    assert tokens == (
        "v3.0.1 gpt-5.4-mini memory_recall openai api "
        "src/alpha_agent/runtime/agent.py c++ c#",
        "v3.0.1",
        "v3",
        "3",
        "0",
        "1",
        "gpt-5.4-mini",
        "gpt",
        "5",
        "4",
        "mini",
        "memory_recall",
        "memory",
        "recall",
        "openai",
        "api",
        "src/alpha_agent/runtime/agent.py",
        "src",
        "alpha_agent",
        "alpha",
        "agent",
        "runtime",
        "agent.py",
        "py",
        "c++",
        "c",
        "c#",
    )


def test_mixed_tokenizer_splits_cjk_and_technical_runs() -> None:
    tokens = tokenize_mixed_text("用户希望在 v3.0.1 里支持 GPT-5.4-mini、memory_recall")

    assert tokens == (
        "用户",
        "希望",
        "在",
        "v3.0.1",
        "v3",
        "3",
        "0",
        "1",
        "里",
        "支持",
        "gpt-5.4-mini",
        "gpt",
        "5",
        "4",
        "mini",
        "memory_recall",
        "memory",
        "recall",
    )


def test_mixed_tokenizer_handles_embedded_ascii_in_cjk_text() -> None:
    tokens = tokenize_mixed_text("用户喜欢Python3.12示例")

    assert tokens == (
        "用户",
        "喜欢",
        "python3.12",
        "python3",
        "python",
        "3",
        "12",
        "示例",
    )


def test_mixed_tokenizer_handles_cjk_name_with_technical_role() -> None:
    tokens = tokenize_mixed_text("小六是assistant名字")

    assert tokens == ("小六是", "assistant", "名字")


def test_mixed_tokenizer_uses_optional_jieba_userdict(tmp_path: Path) -> None:
    userdict = tmp_path / "memory_recall_userdict.txt"
    userdict.write_text("小六 10 n\n", encoding="utf-8")

    tokens = tokenize_mixed_text("小六是assistant名字", userdict_path=userdict)

    assert "小六" in tokens
    assert "assistant" in tokens
    assert "名字" in tokens


def test_jieba_wrapper_ignores_missing_userdict() -> None:
    tokens = cut_jieba_text("小六是名字", userdict_path="docs/todo/missing_userdict.txt")

    assert tokens
