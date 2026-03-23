"""Statistical metrics computation for benchmark results."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """Percentile-based latency statistics in milliseconds."""

    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    stdev: float
    min: float
    max: float
    count: int

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencyStats:
        """Compute stats from a list of latency values (ms)."""
        if not samples:
            return cls(
                mean=0.0, median=0.0, p50=0.0, p90=0.0, p95=0.0,
                p99=0.0, stdev=0.0, min=0.0, max=0.0, count=0,
            )
        sorted_s = sorted(samples)
        n = len(sorted_s)

        def _pct(p: float) -> float:
            idx = (p / 100.0) * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return sorted_s[lo] + (sorted_s[hi] - sorted_s[lo]) * (idx - lo)

        return cls(
            mean=statistics.mean(sorted_s),
            median=statistics.median(sorted_s),
            p50=_pct(50),
            p90=_pct(90),
            p95=_pct(95),
            p99=_pct(99),
            stdev=statistics.stdev(sorted_s) if n > 1 else 0.0,
            min=sorted_s[0],
            max=sorted_s[-1],
            count=n,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "median": self.median,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "stdev": self.stdev,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }


@dataclass
class ThroughputStats:
    """Aggregate throughput statistics."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    wall_time_sec: float
    requests_per_sec: float
    tokens_per_sec: float
    mean_tokens_per_request: float

    @classmethod
    def compute(
        cls,
        total_requests: int,
        successful_requests: int,
        total_tokens: int,
        wall_time_sec: float,
    ) -> ThroughputStats:
        rps = successful_requests / max(wall_time_sec, 1e-9)
        tps = total_tokens / max(wall_time_sec, 1e-9)
        mean_tok = total_tokens / max(successful_requests, 1)
        return cls(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=total_requests - successful_requests,
            total_tokens_generated=total_tokens,
            wall_time_sec=wall_time_sec,
            requests_per_sec=rps,
            tokens_per_sec=tps,
            mean_tokens_per_request=mean_tok,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "wall_time_sec": self.wall_time_sec,
            "requests_per_sec": self.requests_per_sec,
            "tokens_per_sec": self.tokens_per_sec,
            "mean_tokens_per_request": self.mean_tokens_per_request,
        }


@dataclass
class ConcurrencyPoint:
    """Throughput measurement at a specific concurrency level."""

    concurrency: int
    tokens_per_sec: float
    requests_per_sec: float
    p95_ttft_ms: float
    p99_ttft_ms: float


@dataclass
class ScenarioMetrics:
    """All metrics for a single scenario run."""

    scenario_name: str
    engine_name: str
    latency: LatencyStats
    throughput: ThroughputStats
    ttft: LatencyStats
    concurrency_sweep: list[ConcurrencyPoint] = field(default_factory=list)
    kv_cache_timeline: list[float] = field(default_factory=list)  # pct over time
    gpu_memory_timeline: list[float] = field(default_factory=list)  # GB over time
    error_rate: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "engine_name": self.engine_name,
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "ttft": self.ttft.to_dict(),
            "concurrency_sweep": [
                {
                    "concurrency": p.concurrency,
                    "tokens_per_sec": p.tokens_per_sec,
                    "requests_per_sec": p.requests_per_sec,
                    "p95_ttft_ms": p.p95_ttft_ms,
                    "p99_ttft_ms": p.p99_ttft_ms,
                }
                for p in self.concurrency_sweep
            ],
            "kv_cache_timeline": self.kv_cache_timeline,
            "gpu_memory_timeline": self.gpu_memory_timeline,
            "error_rate": self.error_rate,
            "extra": self.extra,
        }


def compute_cdf(samples: Sequence[float], n_points: int = 200) -> tuple[list[float], list[float]]:
    """Return (x_values, cumulative_probabilities) for plotting a CDF."""
    if not samples:
        return [], []
    sorted_s = sorted(samples)
    n = len(sorted_s)
    x = sorted_s
    y = [(i + 1) / n for i in range(n)]

    # Downsample to n_points if needed
    if n > n_points:
        step = n / n_points
        indices = [int(i * step) for i in range(n_points)]
        indices[-1] = n - 1
        x = [sorted_s[i] for i in indices]
        y = [(i + 1) / n for i in indices]

    return x, y


def compare_metrics(
    a: ScenarioMetrics,
    b: ScenarioMetrics,
) -> dict[str, object]:
    """
    Compare two ScenarioMetrics (e.g., vLLM vs SGLang).
    Returns dict of relative deltas (positive = b is better).
    """
    def _delta(va: float, vb: float) -> float:
        if abs(va) < 1e-9:
            return 0.0
        return (va - vb) / va  # positive = a is higher (worse for latency)

    return {
        "engine_a": a.engine_name,
        "engine_b": b.engine_name,
        "ttft_p50_delta_pct": _delta(a.ttft.p50, b.ttft.p50) * 100,
        "ttft_p95_delta_pct": _delta(a.ttft.p95, b.ttft.p95) * 100,
        "total_latency_p95_delta_pct": _delta(a.latency.p95, b.latency.p95) * 100,
        "tokens_per_sec_delta_pct": _delta(b.throughput.tokens_per_sec, a.throughput.tokens_per_sec) * 100,
        "requests_per_sec_delta_pct": _delta(b.throughput.requests_per_sec, a.throughput.requests_per_sec) * 100,
        "kv_cache_mean_a": (
            sum(a.kv_cache_timeline) / len(a.kv_cache_timeline) if a.kv_cache_timeline else 0.0
        ),
        "kv_cache_mean_b": (
            sum(b.kv_cache_timeline) / len(b.kv_cache_timeline) if b.kv_cache_timeline else 0.0
        ),
    }


if __name__ == "__main__":
    import random

    random.seed(42)
    samples = [random.expovariate(1 / 50) for _ in range(1000)]
    stats = LatencyStats.from_samples(samples)
    print("LatencyStats:", stats)

    tp = ThroughputStats.compute(
        total_requests=100,
        successful_requests=98,
        total_tokens=19600,
        wall_time_sec=30.0,
    )
    print("ThroughputStats:", tp)

    x, y = compute_cdf(samples)
    print(f"CDF points: {len(x)}, last x={x[-1]:.1f}ms, last y={y[-1]:.3f}")
    print("Smoke test passed.")
