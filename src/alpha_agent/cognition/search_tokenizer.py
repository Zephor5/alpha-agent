"""Deterministic search tokenization for mixed CJK and technical text."""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

_TECH_STRUCTURE_CHARS = frozenset("_-./:@#+")
_TECH_BOUNDARY_CHARS = frozenset("/:@#+")
_TECH_INNER_CHARS = frozenset("_-.:@#+")


def cut_jieba_text(
    text: object,
    *,
    userdict_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Cut CJK text through the project's jieba wrapper."""

    tokenizer = _jieba_tokenizer(_existing_userdict_key(userdict_path))
    tokens: list[str] = []
    for raw_token in tokenizer.cut(str(text), HMM=True):
        token = _normalize_token(raw_token)
        if _is_meaningful_token(token):
            tokens.append(token)
    return tuple(tokens)


def tokenize_mixed_text(
    text: object,
    *,
    userdict_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Tokenize text into stable CJK, technical, and derived search terms."""

    tokens: list[str] = []
    seen: set[str] = set()
    for kind, value in _iter_runs(str(text)):
        if kind == "cjk":
            run_tokens = cut_jieba_text(value, userdict_path=userdict_path)
        else:
            run_tokens = _technical_tokens(value)
        for token in run_tokens:
            if token not in seen:
                seen.add(token)
                tokens.append(token)
    return tuple(tokens)


@lru_cache(maxsize=16)
def _jieba_tokenizer(userdict_key: str | None) -> Any:
    jieba = import_module("jieba")
    jieba.setLogLevel(logging.WARNING)
    tokenizer = jieba.Tokenizer()
    if userdict_key is not None:
        tokenizer.load_userdict(userdict_key)
    return tokenizer


def _existing_userdict_key(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    return str(candidate) if candidate.exists() else None


def _iter_runs(text: str) -> tuple[tuple[str, str], ...]:
    runs: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if _is_cjk_ideograph(char):
            end = index + 1
            while end < len(text) and _is_cjk_ideograph(text[end]):
                end += 1
            runs.append(("cjk", text[index:end]))
            index = end
            continue
        if _is_technical_char(char):
            value, index = _consume_technical_run(text, index)
            if value:
                runs.append(("technical", value))
            continue
        index += 1
    return tuple(runs)


def _consume_technical_run(text: str, start: int) -> tuple[str, int]:
    chars: list[str] = []
    index = start
    while index < len(text):
        char = text[index]
        if _is_technical_char(char):
            chars.append(char)
            index += 1
            continue
        if char.isspace() and chars and _next_non_space_is_technical(text, index + 1):
            chars.append(" ")
            index = _next_non_space_index(text, index + 1)
            continue
        break
    return "".join(chars).strip(), index


def _next_non_space_index(text: str, start: int) -> int:
    index = start
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _next_non_space_is_technical(text: str, start: int) -> bool:
    index = _next_non_space_index(text, start)
    return index < len(text) and _is_technical_char(text[index])


def _technical_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()

    def emit(raw_token: str) -> None:
        token = _normalize_token(raw_token)
        if token and token not in seen and _is_meaningful_token(token):
            seen.add(token)
            tokens.append(token)

    normalized = _normalize_token(value)
    emit(normalized)
    for space_chunk in _split_on_separators(normalized, frozenset()):
        emit(space_chunk)
        for chunk in _split_on_separators(space_chunk, _TECH_BOUNDARY_CHARS):
            emit(chunk)
            for part in _split_on_separators(chunk, _TECH_INNER_CHARS):
                emit(part)
                for group in _embedded_alpha_digit_groups(part):
                    emit(group)
    return tuple(tokens)


def _split_on_separators(value: str, separators: frozenset[str]) -> tuple[str, ...]:
    parts: list[str] = []
    current: list[str] = []
    for char in value:
        if char in separators or char.isspace():
            if current:
                parts.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return tuple(parts)


def _embedded_alpha_digit_groups(value: str) -> tuple[str, ...]:
    groups: list[str] = []
    current: list[str] = []
    current_kind: str | None = None
    for char in value:
        kind = _ascii_alpha_digit_kind(char)
        if kind is None:
            _append_embedded_group(groups, current, current_kind)
            current = []
            current_kind = None
            continue
        if kind != current_kind:
            _append_embedded_group(groups, current, current_kind)
            current = []
            current_kind = kind
        current.append(char)
    _append_embedded_group(groups, current, current_kind)
    return tuple(groups)


def _append_embedded_group(
    groups: list[str],
    chars: list[str],
    kind: str | None,
) -> None:
    if not chars or kind is None:
        return
    value = "".join(chars)
    if kind == "digit" or len(value) > 1:
        groups.append(value)


def _ascii_alpha_digit_kind(char: str) -> str | None:
    if char.isascii() and char.isalpha():
        return "alpha"
    if char.isascii() and char.isdigit():
        return "digit"
    return None


def _normalize_token(value: object) -> str:
    return " ".join(str(value).casefold().strip().split())


def _is_meaningful_token(token: str) -> bool:
    return any(char.isalnum() or _is_cjk_ideograph(char) for char in token)


def _is_technical_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in _TECH_STRUCTURE_CHARS)


def _is_cjk_ideograph(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0x2CEB0 <= codepoint <= 0x2EBEF
        or 0x30000 <= codepoint <= 0x323AF
    )
