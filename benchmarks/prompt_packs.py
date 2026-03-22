"""Helpers for loading benchmark prompt packs from the repo-local prompts/ directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import cycle, islice
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

JSONL_PACKS = {
    "short_chat": "short_chat.jsonl",
    "long_generation": "long_generation.jsonl",
    "long_context": "long_context.jsonl",
    "structured_json": "structured_json.jsonl",
    "reasoning": "reasoning.jsonl",
}


@dataclass(frozen=True)
class PromptRecord:
    id: str
    category: str
    prompt: str
    target_tokens: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    schema: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptRecord":
        return cls(
            id=str(data["id"]),
            category=str(data.get("category", "unknown")),
            prompt=str(data["prompt"]),
            target_tokens=(
                int(data["target_tokens"]) if data.get("target_tokens") is not None else None
            ),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            schema=(str(data["schema"]) if data.get("schema") is not None else None),
        )


@dataclass(frozen=True)
class SharedPrefixPack:
    id: str
    category: str
    shared_prefix: str
    suffixes: tuple[str, ...]
    target_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedPrefixPack":
        return cls(
            id=str(data["id"]),
            category=str(data.get("category", "shared_prefix")),
            shared_prefix=str(data["shared_prefix"]),
            suffixes=tuple(str(s) for s in data["suffixes"]),
            target_tokens=(
                int(data["target_tokens"]) if data.get("target_tokens") is not None else None
            ),
        )


def default_prompt_pack_for_scenario(scenario_name: str) -> str:
    mapping = {
        "single_request_latency": "short_chat",
        "throughput_ramp": "long_generation",
        "long_context_stress": "long_context",
        "prefix_sharing_benefit": "shared_prefix",
        "structured_generation_speed": "structured_json",
    }
    return mapping.get(scenario_name, "short_chat")


def list_prompt_packs() -> list[str]:
    return sorted([*JSONL_PACKS.keys(), "shared_prefix"])


def load_prompt_pack(pack_name: str, prompts_dir: Path = PROMPTS_DIR) -> list[PromptRecord]:
    if pack_name == "shared_prefix":
        raise ValueError("shared_prefix is not a JSONL pack; use load_shared_prefix_pack()")

    try:
        rel_path = JSONL_PACKS[pack_name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt pack: {pack_name}") from exc

    path = prompts_dir / rel_path
    records: list[PromptRecord] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(PromptRecord.from_dict(json.loads(line)))
    if not records:
        raise ValueError(f"Prompt pack {pack_name} is empty")
    return records


def cycle_prompt_pack(
    pack_name: str,
    count: int,
    prompts_dir: Path = PROMPTS_DIR,
) -> list[PromptRecord]:
    records = load_prompt_pack(pack_name, prompts_dir=prompts_dir)
    return list(islice(cycle(records), count))


def load_shared_prefix_pack(prompts_dir: Path = PROMPTS_DIR) -> SharedPrefixPack:
    path = prompts_dir / "shared_prefix.json"
    return SharedPrefixPack.from_dict(json.loads(path.read_text()))


def load_schema(schema_name: str, prompts_dir: Path = PROMPTS_DIR) -> dict[str, Any]:
    schema_path = prompts_dir / "schemas" / f"{schema_name}.json"
    return json.loads(schema_path.read_text())
