"""
High-concurrency async iterator with limited in-flight tasks.

Notes
-----
- Accepts either a sync iterable or an async iterable of awaitables.
- Also accepts thunks: `Callable[[], Awaitable[T]]`, so awaitables can be
  constructed lazily at scheduling time.
- Keeps the pipeline full *before* yielding results, so a slow consumer does
  not unnecessarily reduce concurrency.
- If the consumer is slow, completed results still accumulate in memory; a
  queue-based design would handle backpressure more explicitly.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Sized
from typing import TypeVar, cast, overload

from tqdm.auto import tqdm

T = TypeVar("T")

type AwaitableLike[T] = Awaitable[T] | Callable[[], Awaitable[T]]
type Source[T] = Iterable[AwaitableLike[T]] | AsyncIterable[AwaitableLike[T]]

_EXHAUSTED = object()


@overload
def _materialize[T](item: Awaitable[T]) -> Awaitable[T]: ...
@overload
def _materialize[T](item: Callable[[], Awaitable[T]]) -> Awaitable[T]: ...


def _materialize[T](item: AwaitableLike[T]) -> Awaitable[T]:
    """
    Turn an awaitable-or-thunk into an awaitable.

    The cast is here because ty does not yet fully narrow `callable(...)`
    via TypeIs/TypeGuard-style reasoning.
    """
    if callable(item):
        thunk = cast(Callable[[], Awaitable[T]], item)
        return thunk()
    return item


def _spawn[T](item: AwaitableLike[T]) -> asyncio.Future[T]:
    """
    Normalize to a scheduled Future.

    - Leaves existing Future/Task objects alone.
    - Wraps coroutines / other awaitables as needed.
    """
    return asyncio.ensure_future(_materialize(item))


async def _sync_to_async_iter[T](src: Iterable[AwaitableLike[T]]) -> AsyncIterator[AwaitableLike[T]]:
    """Wrap a synchronous iterable as an async iterator."""
    for item in src:
        yield item


async def limited_as_completed[T](
    aws: Source[T],
    *,
    in_flight: int,
    return_exceptions: bool = False,
    cancel_on_error: bool = False,
    show_progress: bool = True,
    length_hint: int | None = None,
) -> AsyncIterator[T | BaseException]:
    """
    Yield results as tasks complete, with at most `in_flight` scheduled at once.
    """
    if in_flight <= 0:
        raise ValueError("in_flight must be a positive integer")

    total = length_hint
    if total is None and isinstance(aws, Sized):
        total = len(aws)

    pbar = None
    if show_progress and tqdm is not None:
        smoothing = 0.1 if not total else max(0.01, min(0.3, 25 / total))
        pbar = tqdm(total=total, unit="task", smoothing=smoothing)

    if isinstance(aws, AsyncIterable):
        src = aiter(aws)
    else:
        src = _sync_to_async_iter(aws)

    pending: set[asyncio.Future[T]] = set()

    async def refill() -> None:
        while len(pending) < in_flight:
            item = await anext(src, _EXHAUSTED)
            if item is _EXHAUSTED:
                return
            pending.add(_spawn(item))  # ty:ignore[invalid-argument-type]

    try:
        await refill()

        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Refill before yielding, so a slow consumer doesn't reduce parallelism.
            await refill()

            if pbar is not None:
                pbar.update(len(done))

            for fut in done:
                try:
                    yield fut.result()
                except BaseException as exc:
                    if cancel_on_error or not return_exceptions:
                        # Mark sibling exceptions as retrieved so asyncio
                        # doesn't warn about un-retrieved task exceptions.
                        for other in done:
                            if other is not fut:
                                with contextlib.suppress(BaseException):
                                    other.exception()
                        raise
                    yield exc

    finally:
        if pbar is not None:
            pbar.close()

        for fut in pending:
            fut.cancel()

        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
