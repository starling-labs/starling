"""Milhouse search backend (stub).

The concrete implementation — a hybrid of ClickHouse (entity filtering,
normalization, metadata) and Milvus (vector search) — has been removed for
the public code release. The class signature below preserves the public
interface so the rest of the codebase still type-checks; instantiation
raises ``NotImplementedError``.

Supply your own ``VectorSearchBase`` subclass (and corresponding
``EntityNormalizer``) to run the agents end-to-end. See README.
"""

from __future__ import annotations

from collections.abc import Sequence

from norm_toolkit.models import ConceptInfo

from starling.infra.ch_client import CHClient
from starling.infra.entities import EntityCNF, EntityType
from starling.infra.entity_filter import FilterScope
from starling.infra.normalizer_base import EntityNormalizer
from starling.infra.papers import PaperStore
from starling.infra.vector_search_base import (
    PaperSearchResult,
    VectorSearchBase,
)

_STUB_MESSAGE = (
    "MilhouseVectorSearch implementation is not included in the public code release. "
    "Provide your own VectorSearchBase subclass."
)


class MilhouseVectorSearch(VectorSearchBase):
    """CH + Milvus hybrid backend for VectorSearchBase (stub)."""

    def __init__(
        self,
        ch_client: CHClient,
        *,
        milvus_uri: str | None = None,
        milvus_token: str | None = None,
        collection_name: str = "emb_5_3_full",
        table_name: str = "emb_5_3_full",
        paper_store: PaperStore | None = None,
        normalizer: EntityNormalizer | None = None,
        search_list: int = 200,
    ) -> None:
        raise NotImplementedError(_STUB_MESSAGE)

    async def search(
        self,
        query: str | None,
        top_k: int,
        *,
        pmids: Sequence[str] | None = None,
        entity_groups: EntityCNF | None = None,
        scope: FilterScope = FilterScope.paragraph,
        include_content: bool = False,
        expand_entities: bool = True,
        negate_entity_filter: bool = False,
        stratify_by_pmid: bool = False,
    ) -> list[PaperSearchResult]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def score_pmids(
        self,
        query: str,
        pmids: Sequence[str],
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def count_and_sample(
        self,
        entity_groups: EntityCNF,
        sample_size: int = 10,
        *,
        scope: FilterScope = FilterScope.paragraph,
        expand_entities: bool = True,
        include_content: bool = False,
        use_expanded_bitmaps: bool = False,
        expansion_depth: int = 4,
    ) -> tuple[int, list[PaperSearchResult]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def get_concept_info(self, concept_ids: Sequence[str]) -> dict[str, ConceptInfo]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def fetch_paragraph_entities(
        self,
        targets: Sequence[tuple[str, int]],
    ) -> list[dict[str, object]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def fetch_paragraph_entity_mentions(
        self,
        targets: Sequence[tuple[str, int]],
    ) -> list[dict[str, object]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def count_entity_mentions(
        self,
        entity_type: EntityType,
        min_mentions: int = 1,
        pmids: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        raise NotImplementedError(_STUB_MESSAGE)

    async def fetch_all_entity_hits(
        self,
        entity_groups: EntityCNF,
        *,
        scope: FilterScope = FilterScope.paragraph,
        pmids: Sequence[str] | None = None,
        expand_entities: bool = True,
    ) -> set[str] | set[tuple[str, int]]:
        """Materialize all unique hits for an entity CNF at the requested scope."""
        raise NotImplementedError(_STUB_MESSAGE)

    async def close(self) -> None:
        raise NotImplementedError(_STUB_MESSAGE)
