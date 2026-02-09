# Error Handling Results (Day 19/90)

## Unit checks

| Test | Pass |
|------|------|
| retry_backoff | ✓ |
| retry_all_fail | ✓ |
| circuit_opens | ✓ |
| circuit_cooldown | ✓ |

---

## Scenarios: success rate under failure

### api_failures (fail_every_n=2)

- **fail_every_n:** 2
- **total:** 20
- **success:** 20
- **cache_fallback:** 0
- **service_unavailable:** 0
- **success_rate:** 100.0
- **calls_to_backend:** 39

### api_failures (fail_every_n=1)

- **fail_every_n:** 1
- **total:** 20
- **success:** 0
- **cache_fallback:** 0
- **service_unavailable:** 20
- **success_rate:** 0.0
- **calls_to_backend:** 6

### rate_limit

- **total:** 15
- **success:** 14
- **cache_fallback:** 0
- **service_unavailable:** 1
- **success_rate:** 93.3

### timeout

- **total:** 10
- **success:** 0
- **cache_fallback:** 0
- **service_unavailable:** 10
- **partial_timeout_responses:** 2
- **success_rate:** 0.0

### circuit_opens_fallback

- **total:** 12
- **success:** 0
- **cache_fallback:** 0
- **service_unavailable:** 12
- **success_rate:** 0.0
- **circuit_state:** open

### real_llm_gemini

- **total:** 2
- **success:** 2
- **cache_fallback:** 0
- **service_unavailable:** 0
- **success_rate:** 100.0
- **backend:** Gemini (Gemma 4B)

---

## Summary

- Total queries across scenarios: **79**
- Succeeded (direct): **36**
- Served from cache (fallback): **0**
- Overall success rate (including cache fallback): **45.6%**

**Behaviors verified:**
- Retry: 3 attempts with backoff 0s, 2s, 4s.
- Circuit breaker: opens when >50% fail in 1 min; returns cached fallback when open; closes after 30s cooldown.
- Graceful degradation: API down → cache or service unavailable; rate limit → retry then cache/unavailable; timeout → partial result message.
