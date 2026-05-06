import os

from logfire import ConsoleOptions, LevelName

# Shut gRPC up on Mac
os.environ["GRPC_VERBOSITY"] = "NONE"

import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast

import logfire

P = ParamSpec("P")
T = TypeVar("T")


def setup_logging(
    logfire_min_level: LevelName = "info", console: ConsoleOptions | Literal[False] | None = None
) -> None:
    logfire.configure(min_level=logfire_min_level, console=console)
    logfire.instrument_pydantic_ai()

    logging.getLogger("httpx").setLevel(logging.WARNING)


def logfire_span(span_name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap a function body in a logfire span so we don't need to indent everywhere."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def asyncgen_wrapper(*args: P.args, **kwargs: P.kwargs) -> AsyncGenerator[Any]:
                with logfire.span(span_name):
                    async for item in func(*args, **kwargs):
                        yield item

            return cast(Callable[P, T], asyncgen_wrapper)

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                with logfire.span(span_name):
                    return await cast(Callable[P, Awaitable[T]], func)(*args, **kwargs)

            return cast(Callable[P, T], async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with logfire.span(span_name):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator
