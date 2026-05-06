"""ClickHouse client (stub).

The concrete implementation has been removed for the public code release —
running the agents against a live corpus requires rebuilding this layer
against your own ClickHouse instance. See README for details.
"""

from typing import Any

_STUB_MESSAGE = (
    "ClickHouse client implementation is not included in the public code release. "
    "Provide your own implementation that satisfies the same interface."
)


class CHClient:
    """Async ClickHouse client (stub).

    Concrete implementations should expose:
      - ``query_async(sql, parameters=None, settings=None) -> QueryResult``
      - ``close() -> None``

    where ``QueryResult.result_rows`` is an iterable of row tuples.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    async def query_async(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError(_STUB_MESSAGE)

    async def close(self) -> None:
        raise NotImplementedError(_STUB_MESSAGE)


async def build_clickhouse_engine(max_concurrent: int = 20) -> CHClient:
    raise NotImplementedError(_STUB_MESSAGE)
