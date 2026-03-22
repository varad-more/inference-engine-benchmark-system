"""Tests for benchmarks/prompt_packs.py."""

from __future__ import annotations

from benchmarks.prompt_packs import (
    cycle_prompt_pack,
    list_prompt_packs,
    load_prompt_pack,
    load_schema,
    load_shared_prefix_pack,
)


def test_list_prompt_packs_has_expected() -> None:
    packs = list_prompt_packs()
    for expected in [
        "short_chat",
        "long_generation",
        "long_context",
        "structured_json",
        "reasoning",
        "shared_prefix",
    ]:
        assert expected in packs


def test_load_prompt_pack_short_chat() -> None:
    records = load_prompt_pack("short_chat")
    assert len(records) >= 1
    assert records[0].category == "short_chat"
    assert records[0].prompt


def test_cycle_prompt_pack() -> None:
    records = cycle_prompt_pack("short_chat", 7)
    assert len(records) == 7
    assert all(r.id for r in records)


def test_load_shared_prefix_pack() -> None:
    pack = load_shared_prefix_pack()
    assert pack.category == "shared_prefix"
    assert pack.shared_prefix
    assert len(pack.suffixes) >= 1


def test_load_schema() -> None:
    schema = load_schema("entity_sentiment_v1")
    assert schema["type"] == "object"
    assert "entities" in schema["properties"]
