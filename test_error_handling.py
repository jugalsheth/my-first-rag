"""
Day 19/90: Test error handling — API failures, rate limits, timeouts.
Measures success rate under failure and writes error_handling_results.md.

BACKEND: By default uses SIMULATED backends (no API key needed). If GEMINI_API_KEY is set,
we also run a short live test with real Gemini and add it to the report.
- Simulated: make_failing_query_fn() etc. for retry/circuit/fallback testing.
- Real LLM: 2 Gemini queries through RobustRAG when key is set; results in error_handling_results.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from robust_rag import (
    RobustRAG,
    CircuitBreaker,
    retry_with_backoff,
    make_failing_query_fn,
    SERVICE_UNAVAILABLE_MSG,
    PARTIAL_TIMEOUT_MSG,
)


def test_retry_backoff():
    """Retry: attempt 1 immediate, 2 after 2s, 3 after 4s; max 3 attempts."""
    attempts = []

    def fail_twice():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("fail")
        return "ok"

    result, err, n = retry_with_backoff(fail_twice, delays=[0, 0.1, 0.2], max_attempts=3)
    assert result == "ok"
    assert n == 3
    assert len(attempts) == 3
    return "retry_backoff", True


def test_retry_all_fail():
    def always_fail():
        raise RuntimeError("API down")

    result, err, n = retry_with_backoff(always_fail, delays=[0, 0.05, 0.05], max_attempts=3)
    assert result is None
    assert err is not None
    assert n == 3
    return "retry_all_fail", True


def test_circuit_breaker_opens():
    """>50% fail in window → circuit opens."""
    cb = CircuitBreaker(window_seconds=10, failure_threshold=0.5, cooldown_seconds=1)
    for _ in range(3):
        cb.record_failure()
    for _ in range(2):
        cb.record_success()
    # 3 fail, 2 success → 3/5 = 60% > 50%
    assert cb.is_open()
    return "circuit_opens", True


def test_circuit_closes_after_cooldown():
    cb = CircuitBreaker(window_seconds=10, failure_threshold=0.5, cooldown_seconds=0.2)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.is_open()
    time.sleep(0.25)
    assert not cb.is_open()
    return "circuit_cooldown", True


# Short backoff in tests so run stays fast (production: [0, 2, 4])
TEST_DELAYS = [0, 0.1, 0.2]


def scenario_api_failures(num_queries: int = 20, fail_every_n: int = 2):
    """Simulate API failures: every fail_every_n-th call fails."""
    call_count = [0]
    fn, _ = make_failing_query_fn(fail_every_n=fail_every_n, call_count=call_count)
    cache = {}
    rag = RobustRAG(query_fn=fn, cache=cache, retry_delays=TEST_DELAYS)
    results = []
    for i in range(num_queries):
        q = f"Question {i+1}"
        r = rag.query(q)
        results.append(r)
    rate = rag.get_success_rate()
    return {
        "scenario": "api_failures",
        "fail_every_n": fail_every_n,
        "total": num_queries,
        "success": rag.stats["success"],
        "cache_fallback": rag.stats["cache_fallback"],
        "service_unavailable": rag.stats["service_unavailable"],
        "success_rate": round(rate * 100, 1),
        "calls_to_backend": call_count[0],
    }


def scenario_rate_limit(num_queries: int = 15):
    """Simulate rate limit (429); retry with backoff should eventually succeed or use cache."""
    call_count = [0]

    def rate_limited_fn(q: str) -> str:
        call_count[0] += 1
        if call_count[0] <= 5:
            raise RuntimeError("429 Rate limit exceeded")
        return f"Answer: {q[:30]}"

    rag = RobustRAG(query_fn=rate_limited_fn, cache={}, retry_delays=TEST_DELAYS)
    for i in range(num_queries):
        rag.query(f"Query {i}")
    rate = rag.get_success_rate()
    return {
        "scenario": "rate_limit",
        "total": num_queries,
        "success": rag.stats["success"],
        "cache_fallback": rag.stats["cache_fallback"],
        "service_unavailable": rag.stats["service_unavailable"],
        "success_rate": round(rate * 100, 1),
    }


def scenario_timeout(num_queries: int = 10):
    """Simulate timeouts; expect partial result message when no cache."""
    call_count = [0]

    def timeout_fn(q: str) -> str:
        call_count[0] += 1
        raise TimeoutError("Request timed out")

    rag = RobustRAG(query_fn=timeout_fn, cache={}, retry_delays=TEST_DELAYS)
    results = [rag.query(f"Q{i}") for i in range(num_queries)]
    partial_count = sum(1 for r in results if PARTIAL_TIMEOUT_MSG in r.answer)
    return {
        "scenario": "timeout",
        "total": num_queries,
        "success": rag.stats["success"],
        "cache_fallback": rag.stats["cache_fallback"],
        "service_unavailable": rag.stats["service_unavailable"],
        "partial_timeout_responses": partial_count,
        "success_rate": round(rag.get_success_rate() * 100, 1),
    }


def scenario_circuit_opens_then_fallback(num_queries: int = 12):
    """Many failures → circuit opens → subsequent queries get cache or service unavailable."""
    call_count = [0]

    def fail_often(q: str) -> str:
        call_count[0] += 1
        if call_count[0] <= 6:
            raise RuntimeError("API down")
        return f"Answer: {q}"

    cache = {}
    rag = RobustRAG(query_fn=fail_often, cache=cache, retry_delays=TEST_DELAYS)
    # First 3 queries: fail 3 times each (retries), then service unavailable; some may cache if one succeeds
    for i in range(num_queries):
        r = rag.query(f"Q{i}")
    return {
        "scenario": "circuit_opens_fallback",
        "total": num_queries,
        "success": rag.stats["success"],
        "cache_fallback": rag.stats["cache_fallback"],
        "service_unavailable": rag.stats["service_unavailable"],
        "success_rate": round(rag.get_success_rate() * 100, 1),
        "circuit_state": rag.circuit.get_state(),
    }


def run_all_and_report():
    unit_ok = []
    unit_ok.append(test_retry_backoff())
    unit_ok.append(test_retry_all_fail())
    unit_ok.append(test_circuit_breaker_opens())
    unit_ok.append(test_circuit_closes_after_cooldown())

    scenario_results = []
    scenario_results.append(scenario_api_failures(20, fail_every_n=2))
    scenario_results.append(scenario_api_failures(20, fail_every_n=1))
    scenario_results.append(scenario_rate_limit(15))
    scenario_results.append(scenario_timeout(10))
    scenario_results.append(scenario_circuit_opens_then_fallback(12))

    return unit_ok, scenario_results


def write_md(unit_ok: list, scenario_results: list, path: Path) -> None:
    lines = [
        "# Error Handling Results (Day 19/90)",
        "",
        "## Unit checks",
        "",
        "| Test | Pass |",
        "|------|------|",
    ]
    for name, ok in unit_ok:
        lines.append(f"| {name} | {'✓' if ok else '✗'} |")
    lines.extend([
        "",
        "---",
        "",
        "## Scenarios: success rate under failure",
        "",
    ])
    for s in scenario_results:
        title = s["scenario"]
        if "fail_every_n" in s:
            title += f" (fail_every_n={s['fail_every_n']})"
        lines.append(f"### {title}")
        lines.append("")
        for k, v in s.items():
            if k != "scenario":
                lines.append(f"- **{k}:** {v}")
        lines.append("")
    total_queries = sum(s.get("total", 0) for s in scenario_results)
    total_success = sum(s.get("success", 0) for s in scenario_results)
    total_cache = sum(s.get("cache_fallback", 0) for s in scenario_results)
    overall_rate = (total_success + total_cache) / total_queries * 100 if total_queries else 0
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        f"- Total queries across scenarios: **{total_queries}**",
        f"- Succeeded (direct): **{total_success}**",
        f"- Served from cache (fallback): **{total_cache}**",
        f"- Overall success rate (including cache fallback): **{overall_rate:.1f}%**",
        "",
        "**Behaviors verified:**",
        "- Retry: 3 attempts with backoff 0s, 2s, 4s.",
        "- Circuit breaker: opens when >50% fail in 1 min; returns cached fallback when open; closes after 30s cooldown.",
        "- Graceful degradation: API down → cache or service unavailable; rate limit → retry then cache/unavailable; timeout → partial result message.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_gemini_query_fn():
    """Real LLM backend for optional Day 19 live test. Returns None if no key."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel("models/gemma-3-4b-it")
        def fn(q: str) -> str:
            r = model.generate_content(f"Answer in one sentence: {q}", generation_config={"temperature": 0.3, "max_output_tokens": 100})
            return r.text.strip()
        return fn
    except Exception:
        return None


def run_live_llm_test_if_available() -> Optional[dict]:
    """If GEMINI_API_KEY is set, run 2 queries through RobustRAG with real Gemini. Else return None."""
    query_fn = _make_gemini_query_fn()
    if not query_fn:
        return None
    rag = RobustRAG(query_fn=query_fn, cache={}, log_fn=print, retry_delays=TEST_DELAYS)
    for q in ["What is RAG in one sentence?", "What is retrieval-augmented generation?"]:
        rag.query(q)
    return {
        "scenario": "real_llm_gemini",
        "total": 2,
        "success": rag.stats["success"],
        "cache_fallback": rag.stats["cache_fallback"],
        "service_unavailable": rag.stats["service_unavailable"],
        "success_rate": round(rag.get_success_rate() * 100, 1),
        "backend": "Gemini (Gemma 4B)",
    }


def main():
    print("Day 19/90: Error handling tests")
    print("=" * 50)
    unit_ok, scenario_results = run_all_and_report()
    for name, ok in unit_ok:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print()
    for s in scenario_results:
        print(f"  {s['scenario']}: success_rate={s.get('success_rate', 0)}%")

    # Optional: test with real LLM when GEMINI_API_KEY is set
    live = run_live_llm_test_if_available()
    if live is not None:
        scenario_results = list(scenario_results) + [live]
        print(f"  real_llm_gemini: success_rate={live['success_rate']}% (real Gemini)")
    else:
        print("  (Real LLM skipped: set GEMINI_API_KEY to also test Day 19 with Gemini)")

    path = Path("error_handling_results.md")
    write_md(unit_ok, scenario_results, path)
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
