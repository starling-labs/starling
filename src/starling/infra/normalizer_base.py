"""Abstract base class for entity normalizers.

An entity normalizer resolves free-text terms to canonical concept IDs,
fetches concept metadata, and expands concepts down their is-a hierarchy.
Concrete implementations bind to a specific backend (e.g. ClickHouse).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from norm_toolkit.models import ConceptInfo
from pydantic import BaseModel


class NormMatch(BaseModel):
    global_identifier: str
    pref_name: str
    match_type: str


class EntityNormalizer(ABC):
    """Async entity normalizer interface."""

    @abstractmethod
    async def normalize_many(
        self,
        terms: list[str],
        allow_partial: bool = False,
        top_k: int | None = None,
        ont_top_k: int | None = 3,
    ) -> dict[str, list[NormMatch]]:
        """Normalize a batch of terms to ranked concept matches."""

    @abstractmethod
    async def concept_info(
        self,
        concept_ids: list[str],
    ) -> dict[str, ConceptInfo]:
        """Fetch concept metadata (preferred name, definitions, semantic types)."""

    @abstractmethod
    async def get_narrower_ids_many(
        self,
        global_identifiers: list[str],
        max_depth: int = 4,
        max_ids: int | None = 1000,
    ) -> dict[str, list[str]]:
        """Expand concepts down their is-a hierarchy."""

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by the normalizer."""
