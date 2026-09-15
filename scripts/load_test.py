"""Load and performance benchmarking script for ConsciousAI Journal V2.

Safety Safeguards:
- Rejects non-localhost target URLs by default to prevent accidental production traffic.
- Uses clearly tagged test identifiers and synthetic payload data.
- Measures throughput (RPS), response times (p50, p95, p99), and error rates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from urllib.parse import urlparse

import httpx

ALLOWED_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def validate_target_safety(target_url: str, allow_remote: bool = False) -> None:
    """Validate that the target URL is safe for load testing.

    Raises:
        ValueError: If target URL points to a non-local address without explicit override.
    """
    parsed = urlparse(target_url)
    hostname = parsed.hostname or ""

    if not allow_remote and hostname not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"SAFETY VIOLATION: Target host '{hostname}' is not a local address ({ALLOWED_LOCAL_HOSTS}). "
            "Load testing against remote or production targets is strictly prohibited without --allow-remote."
        )


async def _worker(
    client: httpx.AsyncClient,
    target_url: str,
    endpoint: str,
    semaphore: asyncio.Semaphore,
    results: list[float],
    errors: list[str],
) -> None:
    async with semaphore:
        url = f"{target_url.rstrip('/')}{endpoint}"
        start = time.perf_counter()
        try:
            resp = await client.get(
                url,
                headers={"X-Request-ID": f"load-test-probe-{time.monotonic()}"},
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            if resp.status_code < 400:
                results.append(elapsed_ms)
            else:
                errors.append(f"HTTP_{resp.status_code}")
        except Exception as exc:
            errors.append(exc.__class__.__name__)


async def run_load_test(
    target_url: str = "http://127.0.0.1:8000",
    endpoint: str = "/health",
    total_requests: int = 100,
    concurrency: int = 10,
    allow_remote: bool = False,
) -> dict:
    """Execute concurrent requests against target endpoint and gather latency metrics."""
    validate_target_safety(target_url, allow_remote)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[float] = []
    errors: list[str] = []

    start_wall = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            asyncio.create_task(_worker(client, target_url, endpoint, semaphore, results, errors))
            for _ in range(total_requests)
        ]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_wall

    p50 = statistics.median(results) if results else 0.0
    p95 = (
        statistics.quantiles(results, n=20)[18]
        if len(results) >= 20
        else (max(results) if results else 0.0)
    )
    p99 = (
        statistics.quantiles(results, n=100)[98]
        if len(results) >= 100
        else (max(results) if results else 0.0)
    )

    summary = {
        "target_url": target_url,
        "endpoint": endpoint,
        "total_requests": total_requests,
        "successful_requests": len(results),
        "failed_requests": len(errors),
        "concurrency": concurrency,
        "total_time_seconds": round(total_time, 3),
        "requests_per_second": round(len(results) / total_time, 1) if total_time > 0 else 0.0,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "errors_sample": errors[:5],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ConsciousAI Journal V2 Load Benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL to test")
    parser.add_argument("--endpoint", default="/health", help="API endpoint path")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Total requests")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Explicitly permit testing against non-local hostnames",
    )
    args = parser.parse_args()

    try:
        summary = asyncio.run(
            run_load_test(
                target_url=args.url,
                endpoint=args.endpoint,
                total_requests=args.requests,
                concurrency=args.concurrency,
                allow_remote=args.allow_remote,
            )
        )
        print(json.dumps(summary, indent=2))
        return 0
    except ValueError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"UNEXPECTED FAILURE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
