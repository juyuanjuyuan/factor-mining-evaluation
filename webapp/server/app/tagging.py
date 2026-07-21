"""Validation helpers for user-managed factor tags."""

from __future__ import annotations

import unicodedata


MAX_TAG_LENGTH = 32
MAX_TAGS_PER_FACTOR = 20


def normalize_tag(value: str) -> str:
    """Return a display-safe tag or raise a clear validation error."""
    if not isinstance(value, str):
        raise ValueError("标签必须是文本")
    tag = unicodedata.normalize("NFKC", value).strip()
    if not tag:
        raise ValueError("标签不能为空")
    if len(tag) > MAX_TAG_LENGTH:
        raise ValueError(f"标签不能超过 {MAX_TAG_LENGTH} 个字符")
    if any(not character.isprintable() for character in tag):
        raise ValueError("标签不能包含不可显示字符")
    return tag


def normalize_tags(values: list[str]) -> list[str]:
    """Normalize a tag list while retaining the user's display casing."""
    if len(values) > MAX_TAGS_PER_FACTOR:
        raise ValueError(f"每个因子最多可设置 {MAX_TAGS_PER_FACTOR} 个标签")
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_tag(value)
        key = tag.casefold()
        if key not in seen:
            normalized.append(tag)
            seen.add(key)
    return normalized
