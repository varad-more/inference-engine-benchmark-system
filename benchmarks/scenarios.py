"""Benchmark scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ScenarioType(StrEnum):
    SINGLE_REQUEST_LATENCY = "single_request_latency"
    THROUGHPUT_RAMP = "throughput_ramp"
    LONG_CONTEXT_STRESS = "long_context_stress"
    PREFIX_SHARING_BENEFIT = "prefix_sharing_benefit"
    STRUCTURED_GENERATION_SPEED = "structured_generation_speed"


@dataclass
class BenchmarkScenario:
    """Base configuration for a benchmark scenario."""

    name: str
    scenario_type: ScenarioType
    description: str = ""
    prompt_pack: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "description": self.description,
            "prompt_pack": self.prompt_pack,
        }


@dataclass
class SingleRequestLatency(BenchmarkScenario):
    """
    50 sequential requests at concurrency=1 with short prompts (~64 tokens).
    Measures P50/P95/P99 TTFT and total latency.
    Isolates pure engine overhead with no contention.
    """

    num_requests: int = 50
    concurrency: int = 1
    prompt_tokens: int = 64
    max_output_tokens: int = 128
    temperature: float = 0.0
    scenario_type: ScenarioType = field(default=ScenarioType.SINGLE_REQUEST_LATENCY, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "single_request_latency"
        if not self.description:
            self.description = (
                f"{self.num_requests} sequential requests, concurrency={self.concurrency}, "
                f"~{self.prompt_tokens} prompt tokens, {self.max_output_tokens} max output tokens."
            )
        if self.prompt_pack is None:
            self.prompt_pack = "short_chat"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "num_requests": self.num_requests,
                "concurrency": self.concurrency,
                "prompt_tokens": self.prompt_tokens,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
        )
        return d


@dataclass
class ThroughputRamp(BenchmarkScenario):
    """
    Sweep concurrency from low to high and measure tokens/sec at each level.
    Identifies the saturation point and max sustainable throughput.
    """

    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    requests_per_level: int = 100
    prompt_tokens: int = 128
    max_output_tokens: int = 256
    temperature: float = 0.0
    scenario_type: ScenarioType = field(default=ScenarioType.THROUGHPUT_RAMP, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "throughput_ramp"
        if not self.description:
            self.description = (
                f"Concurrency sweep {self.concurrency_levels}, "
                f"{self.requests_per_level} requests/level."
            )
        if self.prompt_pack is None:
            self.prompt_pack = "long_generation"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "concurrency_levels": self.concurrency_levels,
                "requests_per_level": self.requests_per_level,
                "prompt_tokens": self.prompt_tokens,
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            }
        )
        return d


@dataclass
class LongContextStress(BenchmarkScenario):
    """
    20 requests with 4096-token prompts.
    Stresses KV cache memory and measures GPU utilisation + OOM behaviour.
    """

    num_requests: int = 20
    concurrency: int = 4
    prompt_tokens: int = 4096
    max_output_tokens: int = 256
    temperature: float = 0.0
    scenario_type: ScenarioType = field(default=ScenarioType.LONG_CONTEXT_STRESS, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "long_context_stress"
        if not self.description:
            self.description = (
                f"{self.num_requests} requests with ~{self.prompt_tokens}-token prompts, "
                f"concurrency={self.concurrency}."
            )
        if self.prompt_pack is None:
            self.prompt_pack = "long_context"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "num_requests": self.num_requests,
                "concurrency": self.concurrency,
                "prompt_tokens": self.prompt_tokens,
                "max_output_tokens": self.max_output_tokens,
            }
        )
        return d


@dataclass
class PrefixSharingBenefit(BenchmarkScenario):
    """
    100 requests sharing a 512-token system prompt with varying 50-token user messages.
    Compares vLLM prefix cache vs SGLang RadixAttention cache hit rates.
    Measures TTFT reduction after cache warm-up.
    """

    num_requests: int = 100
    concurrency: int = 8
    shared_prefix_tokens: int = 512
    user_suffix_tokens: int = 50
    max_output_tokens: int = 128
    temperature: float = 0.0
    scenario_type: ScenarioType = field(default=ScenarioType.PREFIX_SHARING_BENEFIT, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "prefix_sharing_benefit"
        if not self.description:
            self.description = (
                f"{self.num_requests} requests with {self.shared_prefix_tokens}-token shared prefix, "
                f"{self.user_suffix_tokens}-token variable suffix."
            )
        if self.prompt_pack is None:
            self.prompt_pack = "shared_prefix"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "num_requests": self.num_requests,
                "concurrency": self.concurrency,
                "shared_prefix_tokens": self.shared_prefix_tokens,
                "user_suffix_tokens": self.user_suffix_tokens,
                "max_output_tokens": self.max_output_tokens,
            }
        )
        return d


@dataclass
class StructuredGenerationSpeed(BenchmarkScenario):
    """
    200 JSON extraction requests.
    Compares native SGLang constrained decode vs vLLM + Outlines integration.
    """

    num_requests: int = 200
    concurrency: int = 16
    max_output_tokens: int = 150
    temperature: float = 0.0
    json_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "entities": {"type": "array", "items": {"type": "string"}},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            },
            "required": ["entities", "sentiment"],
        }
    )
    scenario_type: ScenarioType = field(
        default=ScenarioType.STRUCTURED_GENERATION_SPEED, init=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "structured_generation_speed"
        if not self.description:
            self.description = (
                f"{self.num_requests} JSON entity-extraction requests, "
                f"concurrency={self.concurrency}."
            )
        if self.prompt_pack is None:
            self.prompt_pack = "structured_json"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "num_requests": self.num_requests,
                "concurrency": self.concurrency,
                "max_output_tokens": self.max_output_tokens,
                "json_schema": self.json_schema,
            }
        )
        return d


# ------------------------------------------------------------------
# Prompt generators
# ------------------------------------------------------------------


def make_short_prompt(target_tokens: int = 64) -> str:
    """Generate a deterministic prompt of approximately target_tokens tokens."""
    words = (
        "The quick brown fox jumps over the lazy dog. "
        "In a distant galaxy far away, scientists discovered new forms of life. "
        "Machine learning models require vast computational resources to train effectively. "
        "The study of linguistics reveals how language shapes human thought and culture. "
    )
    # Repeat until we have enough words (~1.3 tokens/word on average)
    target_words = int(target_tokens / 1.3)
    repeated = (words * (target_words // len(words.split()) + 2)).split()
    return " ".join(repeated[:target_words])


def make_long_prompt(target_tokens: int = 4096) -> str:
    """Generate a long prompt approximating target_tokens tokens."""
    paragraph = (
        "Artificial intelligence and machine learning have transformed how we process "
        "information at scale. Large language models leverage transformer architectures "
        "to understand and generate human-like text across diverse domains including "
        "medicine, law, software engineering, and creative writing. The attention "
        "mechanism, introduced in 'Attention Is All You Need', enables these models "
        "to capture long-range dependencies in text. Modern inference engines like "
        "vLLM and SGLang optimize throughput through techniques such as PagedAttention, "
        "continuous batching, and RadixAttention for prefix sharing. "
    )
    target_words = int(target_tokens / 1.3)
    repeated = (paragraph * (target_words // len(paragraph.split()) + 2)).split()
    return " ".join(repeated[:target_words])


def make_system_prompt(target_tokens: int = 512) -> str:
    """Generate a realistic system prompt of approximately target_tokens tokens."""
    base = (
        "You are an expert AI assistant specializing in data analysis and natural language "
        "processing. Your role is to help users extract structured information from unstructured "
        "text, identify key entities, assess sentiment, and provide actionable insights. "
        "Always respond in valid JSON format when asked for structured output. "
        "Be precise, concise, and accurate. If you are uncertain, say so explicitly. "
        "Do not hallucinate facts or make up information that is not present in the input. "
        "Format numbers clearly and use ISO date formats (YYYY-MM-DD) for dates. "
        "When listing entities, deduplicate and normalise to canonical forms. "
    )
    target_words = int(target_tokens / 1.3)
    repeated = (base * (target_words // len(base.split()) + 2)).split()
    return " ".join(repeated[:target_words])


def make_json_extraction_prompt(text: str) -> str:
    return (
        f"Extract all named entities and the overall sentiment from the following text. "
        f"Respond ONLY with a JSON object with keys 'entities' (list of strings) and "
        f"'sentiment' (one of: positive, negative, neutral).\n\nText: {text}\n\nJSON:"
    )


# Pre-built scenario instances for quick import
SCENARIOS: dict[str, BenchmarkScenario] = {
    "single_request_latency": SingleRequestLatency(name="single_request_latency"),
    "throughput_ramp": ThroughputRamp(name="throughput_ramp"),
    "long_context_stress": LongContextStress(name="long_context_stress"),
    "prefix_sharing_benefit": PrefixSharingBenefit(name="prefix_sharing_benefit"),
    "structured_generation_speed": StructuredGenerationSpeed(name="structured_generation_speed"),
}


if __name__ == "__main__":
    import json

    for name, scenario in SCENARIOS.items():
        print(f"\n{'=' * 60}")
        print(json.dumps(scenario.to_dict(), indent=2))

    short = make_short_prompt(64)
    print(f"\nShort prompt ({len(short.split())} words): {short[:80]}...")

    long_p = make_long_prompt(4096)
    print(f"Long prompt ({len(long_p.split())} words)")

    sys_p = make_system_prompt(512)
    print(f"System prompt ({len(sys_p.split())} words)")

    print("\nSmoke test passed.")
