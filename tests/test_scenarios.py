"""Tests for benchmarks/scenarios.py."""

from __future__ import annotations

import pytest

from benchmarks.scenarios import (
    LongContextStress,
    PrefixSharingBenefit,
    SCENARIOS,
    ScenarioType,
    SingleRequestLatency,
    StructuredGenerationSpeed,
    ThroughputRamp,
    make_json_extraction_prompt,
    make_long_prompt,
    make_short_prompt,
    make_system_prompt,
)


class TestScenarioInstances:
    def test_single_request_latency_defaults(self) -> None:
        s = SingleRequestLatency(name="test")
        assert s.num_requests == 50
        assert s.concurrency == 1
        assert s.prompt_tokens == 64
        assert s.scenario_type == ScenarioType.SINGLE_REQUEST_LATENCY

    def test_throughput_ramp_defaults(self) -> None:
        s = ThroughputRamp(name="test")
        assert s.concurrency_levels == [1, 2, 4, 8, 16, 32, 64]
        assert s.requests_per_level == 100
        assert s.scenario_type == ScenarioType.THROUGHPUT_RAMP

    def test_long_context_stress_defaults(self) -> None:
        s = LongContextStress(name="test")
        assert s.prompt_tokens == 4096
        assert s.num_requests == 20
        assert s.scenario_type == ScenarioType.LONG_CONTEXT_STRESS

    def test_prefix_sharing_benefit_defaults(self) -> None:
        s = PrefixSharingBenefit(name="test")
        assert s.shared_prefix_tokens == 512
        assert s.user_suffix_tokens == 50
        assert s.num_requests == 100
        assert s.scenario_type == ScenarioType.PREFIX_SHARING_BENEFIT

    def test_structured_generation_defaults(self) -> None:
        s = StructuredGenerationSpeed(name="test")
        assert s.num_requests == 200
        assert s.scenario_type == ScenarioType.STRUCTURED_GENERATION_SPEED
        assert "entities" in s.json_schema["properties"]
        assert "sentiment" in s.json_schema["properties"]

    def test_to_dict_contains_scenario_type(self) -> None:
        s = SingleRequestLatency(name="x")
        d = s.to_dict()
        assert d["scenario_type"] == ScenarioType.SINGLE_REQUEST_LATENCY.value
        assert d["num_requests"] == 50

    def test_all_scenarios_in_registry(self) -> None:
        expected = {
            "single_request_latency",
            "throughput_ramp",
            "long_context_stress",
            "prefix_sharing_benefit",
            "structured_generation_speed",
        }
        assert set(SCENARIOS.keys()) == expected


class TestPromptGenerators:
    def test_short_prompt_length(self) -> None:
        p = make_short_prompt(64)
        words = p.split()
        # Rough: target_words = int(64 / 1.3) ≈ 49
        assert 40 <= len(words) <= 70

    def test_long_prompt_length(self) -> None:
        p = make_long_prompt(4096)
        words = p.split()
        # Rough: target_words ≈ 3150
        assert 2500 <= len(words) <= 4000

    def test_system_prompt_length(self) -> None:
        p = make_system_prompt(512)
        words = p.split()
        assert 300 <= len(words) <= 600

    def test_json_extraction_prompt_format(self) -> None:
        text = "Apple CEO Tim Cook announced Q4 results."
        prompt = make_json_extraction_prompt(text)
        assert "entities" in prompt.lower()
        assert "sentiment" in prompt.lower()
        assert text in prompt
        assert prompt.endswith("JSON:")

    def test_prompts_deterministic(self) -> None:
        p1 = make_short_prompt(64)
        p2 = make_short_prompt(64)
        assert p1 == p2
