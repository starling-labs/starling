# Pulled straight from pydantic-ai docs

from httpx import AsyncClient, AsyncHTTPTransport, HTTPStatusError, Limits
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential


def create_retrying_client(max_concurrency: int = 10):
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 500, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    limits = Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )

    # If we don't pass in a base transport, one with default limits will be created
    base_transport = AsyncHTTPTransport(
        limits=limits,
    )

    transport = AsyncTenacityTransport(
        wrapped=base_transport,
        config=RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(fallback_strategy=wait_exponential(multiplier=1, max=60), max_wait=300),
            # Stop after 10 attempts
            stop=stop_after_attempt(10),
            # Re-raise the last exception if all retries fail
            reraise=True,
        ),
        validate_response=should_retry_status,
    )
    return AsyncClient(
        transport=transport,
        http2=True,
        limits=limits,
        timeout=3600,
    )
