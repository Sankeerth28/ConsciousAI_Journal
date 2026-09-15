# Monitoring & Alerting Guide — ConsciousAI Journal V2

This document details observability infrastructure, log correlation, metrics, and alerting policies for ConsciousAI Journal V2.

## 1. Observability Architecture

### Request Correlation (`X-Request-ID`)
Every incoming HTTP request receives or inherits an `X-Request-ID`. This UUID is:
- Attached to the request state.
- Returned in the HTTP response header `X-Request-ID`.
- Included in every structured access log line emitted by `app.access`.

### Privacy-Preserving Access Logging
Access logs record transaction metadata without capturing sensitive user data:
```
request_id=c148286a-7fd3-40f7-bce2-6582531d0442 method=POST path=/api/v1/journals status=201 duration_ms=42.15
```

> [!IMPORTANT]
> The logging middleware strictly excludes request bodies, passwords, JWT tokens, journal contents, and database connection strings to guarantee user confidentiality.

## 2. Health & Readiness Probes

| Endpoint | Probe Type | Check Criteria | Normal Response | Failure Response |
| :--- | :--- | :--- | :--- | :--- |
| `/health` | **Liveness** | Fast process heartbeat (no DB/Redis dependencies) | `200 OK` | Container crash / unhandled event loop hang |
| `/ready` | **Readiness** | PostgreSQL `SELECT 1` + Redis `PING` | `200 OK` | `503 Service Unavailable` with sanitized status |

## 3. Recommended Prometheus / CloudWatch Metrics

1. **HTTP Metrics**:
   - `http_requests_total{status=~"5.."}`: Total server errors.
   - `http_requests_total{status="429"}`: Rate-limited requests.
   - `http_request_duration_seconds{quantile="0.95"}`: 95th percentile latency.
2. **Database Metrics**:
   - `db_pool_connections_in_use`: Current active connections out of `DB_POOL_SIZE + DB_MAX_OVERFLOW`.
   - `db_query_duration_seconds`: Query latency.
3. **Redis Metrics**:
   - `redis_connected_clients`: Active Redis client connections.
   - `redis_memory_used_bytes`: Memory utilization of sliding window keys.

## 4. Alerting Thresholds

- **Critical Alert — Service Unavailable**:
  - Condition: `/ready` returns HTTP 503 for > 3 consecutive checks (30 seconds).
  - Action: Page on-call engineer; check PostgreSQL and Redis cluster health.
- **High Alert — Elevated 5xx Error Rate**:
  - Condition: 5xx error rate exceeds 1% of total traffic over a 5-minute window.
  - Action: Inspect application logs filtering by recent unhandled exceptions.
- **Warning Alert — Elevated Rate Limiting**:
  - Condition: 429 response rate exceeds 5% of traffic.
  - Action: Inspect source IPs for potential denial-of-service or bot activity.
- **Warning Alert — High Latency**:
  - Condition: p95 latency exceeds 500ms on `/api/v1/journals` for > 10 minutes.
  - Action: Inspect database query latency and AI model inference delays.
