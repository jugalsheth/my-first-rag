"""
Day 19/90: Error handling for RAG — retry with backoff, circuit breaker, graceful degradation.

Backend: RobustRAG wraps any query_fn(question) -> answer. It does NOT call Gemini or any API
itself. You pass in the backend (e.g. a function that calls your RAG/Gemini). Tests use
make_failing_query_fn() which is a fake backend (returns or raises) for deterministic testing.
To use with real Gemini: wrap your existing RAG's .query(question) in a lambda and pass it
as query_fn; set log_fn=print to see attempts, cache hits, and circuit state.

Example with real backend and logging:
  from agentic_rag import AgenticRAG
  rag_backend = AgenticRAG()
  def real_query(q): return rag_backend.generate_answer(q, rag_backend.retrieve_chunks(q)[0])
  robust = RobustRAG(query_fn=real_query, cache={}, log_fn=print)
  result = robust.query("What is RAG?")  # logs: [RobustRAG] success after 1 attempt(s)
"""

from __future__ import annotations

import time
import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Backoff delays in seconds: attempt 1 immediate, 2 wait 2s, 3 wait 4s
RETRY_DELAYS = [0, 2, 4]
MAX_ATTEMPTS = 3

# Circuit breaker: >50% fail in 1 min → open; close after 30s cooldown
CIRCUIT_FAILURE_WINDOW_SECONDS = 60
CIRCUIT_FAILURE_THRESHOLD = 0.5
CIRCUIT_COOLDOWN_SECONDS = 30

# Fallback messages
SERVICE_UNAVAILABLE_MSG = "Service temporarily unavailable. Please try again later."
PARTIAL_TIMEOUT_MSG = "Partial results unavailable (request timed out)."


class CircuitBreaker:
    """
    Tracks failures in a time window. If failure rate > 50% in last 1 min, circuit opens.
    Returns cached fallback when open. Closes after 30s cooldown.
    """

    def __init__(
        self,
        window_seconds: float = CIRCUIT_FAILURE_WINDOW_SECONDS,
        failure_threshold: float = CIRCUIT_FAILURE_THRESHOLD,
        cooldown_seconds: float = CIRCUIT_COOLDOWN_SECONDS,
    ):
        self.window_seconds = window_seconds
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failure_times: deque = deque()
        self._success_times: deque = deque()
        self._opened_at: Optional[float] = None

    def record_success(self) -> None:
        self._success_times.append(time.monotonic())

    def record_failure(self) -> None:
        self._failure_times.append(time.monotonic())

    def _trim_old(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._failure_times and self._failure_times[0] < cutoff:
            self._failure_times.popleft()
        while self._success_times and self._success_times[0] < cutoff:
            self._success_times.popleft()

    def is_open(self) -> bool:
        now = time.monotonic()
        if self._opened_at is None:
            self._trim_old()
            total = len(self._failure_times) + len(self._success_times)
            if total < 2:
                return False
            rate = len(self._failure_times) / total
            if rate > self.failure_threshold:
                self._opened_at = now
                return True
            return False
        if now - self._opened_at >= self.cooldown_seconds:
            self._opened_at = None
            return False
        return True

    def get_state(self) -> str:
        return "open" if self.is_open() else "closed"


def retry_with_backoff(
    fn: Callable[[], Any],
    delays: List[float] = RETRY_DELAYS,
    max_attempts: int = MAX_ATTEMPTS,
) -> Tuple[Any, Optional[Exception], int]:
    """
    Call fn up to max_attempts times with delays between attempts.
    Returns (result, error, attempts_used). If all fail, result is None and error is last exception.
    """
    last_error = None
    for attempt in range(max_attempts):
        if attempt > 0 and attempt - 1 < len(delays):
            time.sleep(delays[attempt - 1])
        try:
            out = fn()
            return out, None, attempt + 1
        except Exception as e:
            last_error = e
    return None, last_error, max_attempts


@dataclass
class QueryResult:
    answer: str
    from_cache: bool
    circuit_open: bool
    attempts: int
    error: Optional[str] = None
    degraded: bool = False  # timeout/partial


class RobustRAG:
    """
    Wraps a RAG query function with retry (exponential backoff), circuit breaker,
    and graceful degradation (cached fallback, service unavailable).
    """

    def __init__(
        self,
        query_fn: Callable[[str], str],
        cache: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_delays: Optional[List[float]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            query_fn: (question: str) -> answer: str. Your backend (e.g. RAG or Gemini caller). Can raise for API/rate/timeout.
            cache: optional dict query_key -> answer for fallback.
            timeout_seconds: if set, wrap query in timeout (simulated in tests).
            circuit_breaker: if None, one is created with defaults.
            retry_delays: delays between retries (e.g. [0, 2, 4]). Default RETRY_DELAYS.
            log_fn: if set (e.g. print or logging.info), log each query: attempt, cache hit, circuit, result.
        """
        self.query_fn = query_fn
        self.cache = cache if cache is not None else {}
        self.timeout_seconds = timeout_seconds
        self.circuit = circuit_breaker or CircuitBreaker()
        self.retry_delays = retry_delays if retry_delays is not None else RETRY_DELAYS
        self.log_fn = log_fn
        self.stats = {
            "total": 0,
            "success": 0,
            "cache_fallback": 0,
            "service_unavailable": 0,
            "degraded": 0,
        }

    def _cache_key(self, question: str) -> str:
        return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:24]

    def _get_cached(self, question: str) -> Optional[str]:
        return self.cache.get(self._cache_key(question))

    def _set_cached(self, question: str, answer: str) -> None:
        self.cache[self._cache_key(question)] = answer

    def _log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(f"[RobustRAG] {msg}")

    def query(self, question: str) -> QueryResult:
        self.stats["total"] += 1

        if self.circuit.is_open():
            self._log("circuit open → using cache or service unavailable")
            cached = self._get_cached(question)
            if cached is not None:
                self.stats["cache_fallback"] += 1
                self._log("cache hit (circuit open)")
                return QueryResult(
                    answer=cached,
                    from_cache=True,
                    circuit_open=True,
                    attempts=0,
                    degraded=True,
                )
            self.stats["service_unavailable"] += 1
            self._log("service unavailable (circuit open, no cache)")
            return QueryResult(
                answer=SERVICE_UNAVAILABLE_MSG,
                from_cache=False,
                circuit_open=True,
                attempts=0,
                error="circuit_open",
            )

        def do_query() -> str:
            return self.query_fn(question)

        result, err, attempts = retry_with_backoff(do_query, delays=self.retry_delays)

        if result is not None:
            self.circuit.record_success()
            self._log(f"success after {attempts} attempt(s)")
            self._set_cached(question, result)
            self.stats["success"] += 1
            return QueryResult(
                answer=result,
                from_cache=False,
                circuit_open=False,
                attempts=attempts,
            )

        self.circuit.record_failure()
        self._log(f"all {attempts} attempt(s) failed: {err}")
        cached = self._get_cached(question)
        if cached is not None:
            self.stats["cache_fallback"] += 1
            self._log("cache hit (after failure)")
            return QueryResult(
                answer=cached,
                from_cache=True,
                circuit_open=False,
                attempts=attempts,
                error=str(err) if err else None,
                degraded=True,
            )
        self.stats["service_unavailable"] += 1
        self._log("service unavailable (no cache)")
        # Timeout → partial result message when no cache
        is_timeout = err and isinstance(err, TimeoutError)
        answer = PARTIAL_TIMEOUT_MSG if is_timeout else SERVICE_UNAVAILABLE_MSG
        return QueryResult(
            answer=answer,
            from_cache=False,
            circuit_open=self.circuit.is_open(),
            attempts=attempts,
            error=str(err) if err else None,
            degraded=is_timeout,
        )

    def get_success_rate(self) -> float:
        if self.stats["total"] == 0:
            return 0.0
        return (self.stats["success"] + self.stats["cache_fallback"]) / self.stats["total"]


def make_failing_query_fn(
    fail_every_n: int = 1,
    raise_rate_limit: bool = False,
    raise_timeout: bool = False,
    call_count: Optional[List[int]] = None,
) -> Tuple[Callable[[str], str], List[int]]:
    """
    Returns (query_fn, call_count_list) for testing.
    fail_every_n: 1 = always fail, 2 = every 2nd fails, 0 = never fail.
    raise_rate_limit / raise_timeout: raise specific error type.
    call_count is mutated to count calls.
    """
    if call_count is None:
        call_count = [0]  # type: List[int]

    def fn(question: str) -> str:
        call_count[0] += 1
        n = call_count[0]
        if fail_every_n and n % fail_every_n == 0:
            if raise_rate_limit:
                raise RuntimeError("429 Rate limit exceeded")
            if raise_timeout:
                raise TimeoutError("Request timed out")
            raise RuntimeError("API unavailable")
        return f"Answer for: {question[:50]}"

    return fn, call_count
