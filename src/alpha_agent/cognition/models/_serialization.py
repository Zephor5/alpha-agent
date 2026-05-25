"""Record helpers shared by cognition model dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints


def to_record_value(value: Any) -> Any:
    """Convert dataclass, enum, list, and dict values into JSON-safe records."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_record_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, list):
        return [to_record_value(item) for item in value]
    if isinstance(value, tuple):
        return [to_record_value(item) for item in value]
    if isinstance(value, dict):
        return {str(to_record_value(key)): to_record_value(item) for key, item in value.items()}
    return value


def from_record_value(annotation: Any, value: Any) -> Any:
    """Best-effort reconstruction for Phase 01 typed dataclass records."""

    if value is None:
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {list, tuple} and args:
        return [from_record_value(args[0], item) for item in value]
    if origin is dict and len(args) == 2:
        return {
            from_record_value(args[0], key): from_record_value(args[1], item)
            for key, item in value.items()
        }
    if origin is type(None):
        return None
    if origin in {UnionType, Union} and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        return from_record_value(non_none[0], value) if non_none else value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
        return dataclass_from_record(annotation, value)
    if hasattr(annotation, "__supertype__"):
        return annotation(value)
    return value


def dataclass_to_record(instance: Any) -> dict[str, Any]:
    """Serialize a dataclass instance into a JSON-safe dict."""

    return to_record_value(instance)


def dataclass_from_record[T](cls: type[T], record: dict[str, Any]) -> T:
    """Rebuild a dataclass from a record using field annotations."""

    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name in record:
            kwargs[field.name] = from_record_value(
                type_hints.get(field.name, field.type),
                record[field.name],
            )
    return cls(**kwargs)
