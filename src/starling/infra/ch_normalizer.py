"""ClickHouse-backed EntityNormalizer (stub).

The concrete implementation has been removed for the public code release.
The class signature below documents the interface; supply your own
``EntityNormalizer`` implementation to run the agents end-to-end.
"""

from __future__ import annotations

from typing import Any

from norm_toolkit.models import ConceptInfo

from starling.infra.ch_client import CHClient
from starling.infra.normalizer_base import EntityNormalizer, NormMatch

__all__ = ["CHEntityNormalizer", "NormMatch"]


_STUB_MESSAGE = (
    "CHEntityNormalizer implementation is not included in the public code release. "
    "Provide your own EntityNormalizer subclass."
)


class CHEntityNormalizer(EntityNormalizer):
    """Async EntityNormalizer backed by ClickHouse (stub)."""

    def __init__(self, ch_client: CHClient, **kwargs: Any) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    async def normalize_many(
        self,
        terms: list[str],
        allow_partial: bool = False,
        top_k: int | None = None,
        ont_top_k: int | None = 3,
    ) -> dict[str, list[NormMatch]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def concept_info(
        self,
        concept_ids: list[str],
    ) -> dict[str, ConceptInfo]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def get_narrower_ids_many(
        self,
        global_identifiers: list[str],
        max_depth: int = 4,
        max_ids: int | None = 1000,
    ) -> dict[str, list[str]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def close(self) -> None:
        raise NotImplementedError(_STUB_MESSAGE)
