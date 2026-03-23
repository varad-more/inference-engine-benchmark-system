"""Tests for benchmarks/metrics.py."""

from __future__ import annotations

import pytest

from benchmarks.metrics import (
    LatencyStats,
    ScenarioMetrics,
    ThroughputStats,
    compare_metrics,
    compute_cdf,
)

# ---------------------------------------------------------------------------
# LatencyStats
# ---------------------------------------------------------------------------

class TestLatencyStats:
    def test_empty(self) -> None:
        s = LatencyStats.from_samples([])
        assert s.count == 0
        assert s.mean == 0.0
        assert s.p50 == 0.0

    def test_single(self) -> None:
        s = LatencyStats.from_samples([42.0])
        assert s.count == 1
        assert s.mean == 42.0
        assert s.p50 == 42.0
        assert s.p95 == 42.0
        assert s.stdev == 0.0

    def test_uniform(self) -> None:
        samples = [100.0] * 100
        s = LatencyStats.from_samples(samples)
        assert s.mean == 100.0
        assert s.p50 == 100.0
        assert s.p99 == 100.0
        assert s.stdev == 0.0

    def test_sorted_ordering(self) -> None:
        samples = [50.0, 10.0, 30.0, 90.0, 70.0]
        s = LatencyStats.from_samples(samples)
        assert s.min == 10.0
        assert s.max == 90.0
        assert 30.0 <= s.p50 <= 50.0

    def test_percentiles_ordered(self) -> None:
        import random
        random.seed(0)
        samples = [random.expovariate(1 / 100) for _ in range(500)]
        s = LatencyStats.from_samples(samples)
        assert s.p50 <= s.p90 <= s.p95 <= s.p99
        assert s.min <= s.p50
        assert s.p99 <= s.max

    def test_to_dict_keys(self) -> None:
        s = LatencyStats.from_samples([1.0, 2.0, 3.0])
        d = s.to_dict()
        for key in ["mean", "median", "p50", "p90", "p95", "p99", "stdev", "min", "max", "count"]:
            assert key in d

    def test_large_samples_p99(self) -> None:
        # With 1000 samples in [0, 999], p99 should be near 990
        samples = [float(i) for i in range(1000)]
        s = LatencyStats.from_samples(samples)
        assert 985.0 <= s.p99 <= 999.0


# ---------------------------------------------------------------------------
# ThroughputStats
# ---------------------------------------------------------------------------

class TestThroughputStats:
    def test_basic(self) -> None:
        t = ThroughputStats.compute(
            total_requests=100,
            successful_requests=95,
            total_tokens=9500,
            wall_time_sec=10.0,
        )
        assert t.total_requests == 100
        assert t.successful_requests == 95
        assert t.failed_requests == 5
        assert abs(t.requests_per_sec - 9.5) < 0.01
        assert abs(t.tokens_per_sec - 950.0) < 0.01
        assert abs(t.mean_tokens_per_request - 100.0) < 0.01

    def test_zero_time(self) -> None:
        # Should not divide by zero
        t = ThroughputStats.compute(100, 100, 1000, 0.0)
        assert t.tokens_per_sec > 0

    def test_no_requests(self) -> None:
        t = ThroughputStats.compute(0, 0, 0, 1.0)
        assert t.tokens_per_sec == 0.0
        assert t.mean_tokens_per_request == 0.0

    def test_to_dict_keys(self) -> None:
        t = ThroughputStats.compute(10, 10, 100, 1.0)
        d = t.to_dict()
        for key in ["total_requests", "successful_requests", "tokens_per_sec", "requests_per_sec"]:
            assert key in d


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

class TestComputeCDF:
    def test_empty(self) -> None:
        x, y = compute_cdf([])
        assert x == []
        assert y == []

    def test_single(self) -> None:
        x, y = compute_cdf([42.0])
        assert x == [42.0]
        assert y == [1.0]

    def test_monotone(self) -> None:
        import random
        random.seed(1)
        samples = [random.random() * 1000 for _ in range(200)]
        x, y = compute_cdf(samples)
        assert len(x) == len(y)
        assert all(x[i] <= x[i + 1] for i in range(len(x) - 1))
        assert all(y[i] <= y[i + 1] for i in range(len(y) - 1))
        assert y[-1] == pytest.approx(1.0, abs=0.01)

    def test_downsamples(self) -> None:
        samples = list(range(1000))
        x, y = compute_cdf(samples, n_points=100)
        assert len(x) == 100
        assert len(y) == 100


# ---------------------------------------------------------------------------
# compare_metrics
# ---------------------------------------------------------------------------

class TestCompareMetrics:
    def _make_metrics(self, ttft_p95: float, tps: float, engine: str) -> ScenarioMetrics:
        latency = LatencyStats.from_samples([50.0, 60.0, 70.0, 80.0, 120.0])
        ttft = LatencyStats.from_samples([10.0, 12.0, ttft_p95, ttft_p95 + 5, ttft_p95 + 10])
        throughput = ThroughputStats.compute(100, 100, int(tps * 10), 10.0)
        return ScenarioMetrics(
            scenario_name="test",
            engine_name=engine,
            latency=latency,
            throughput=throughput,
            ttft=ttft,
        )

    def test_structure(self) -> None:
        a = self._make_metrics(ttft_p95=50.0, tps=1000.0, engine="vllm")
        b = self._make_metrics(ttft_p95=40.0, tps=1200.0, engine="sglang")
        delta = compare_metrics(a, b)
        assert "engine_a" in delta
        assert "engine_b" in delta
        assert "ttft_p95_delta_pct" in delta
        assert "tokens_per_sec_delta_pct" in delta

    def test_delta_direction(self) -> None:
        # a has higher TTFT → positive delta (a is worse)
        a = self._make_metrics(ttft_p95=100.0, tps=500.0, engine="vllm")
        b = self._make_metrics(ttft_p95=50.0, tps=1000.0, engine="sglang")
        delta = compare_metrics(a, b)
        # ttft_p95_delta_pct = (a.p95 - b.p95) / a.p95 * 100 should be positive
        assert delta["ttft_p95_delta_pct"] > 0

    def test_equal_engines(self) -> None:
        a = self._make_metrics(ttft_p95=50.0, tps=1000.0, engine="vllm")
        b = self._make_metrics(ttft_p95=50.0, tps=1000.0, engine="sglang")
        delta = compare_metrics(a, b)
        assert abs(delta["ttft_p95_delta_pct"]) < 1.0
