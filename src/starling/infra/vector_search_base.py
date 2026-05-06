from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from norm_toolkit.models import ConceptInfo
from pydantic import BaseModel, Field

from starling.infra.entities import EntityCNF, EntityType
from starling.infra.entity_filter import FilterScope
from starling.infra.papers import Paper, PaperStore

# Allowed embedding table names (shared across implementations)
ALLOWED_TABLES = [
    "emb_5_3_full",
]


class PaperSearchResult(BaseModel):
    """Result from a vector/entity search.

    Attributes:
        pmid: PubMed ID of the paper.
        title: Paper title (may be None if metadata unavailable).
        abstract: Paper abstract (may be None).
        journal: Journal name (may be None).
        chunks: List of chunk/paragraph indices matching the query.
            For vector search, these are embedding chunk indices.
            For entity search with paragraph scope, contains matched paragraph index.
        matched_paragraphs: Paragraph indices satisfying entity filter.
        vec_sim: Vector similarity score (0.0 for entity-only searches).
        content: Full paper content if include_content=True was requested.
    """

    pmid: str
    title: str | None
    abstract: str | None
    journal: str | None

    chunks: list[int]
    matched_paragraphs: list[int] = Field(default_factory=list)

    vec_sim: float

    content: Paper | None = None


class VectorSearchBase(ABC):
    """Abstract base class for vector search implementations.

    Scope Options:
        - "paper": Paper-level filtering
        - "chunk": Chunk-level filtering
        - "paragraph": Paragraph-level CNF satisfaction
    """

    paper_store: PaperStore

    @abstractmethod
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
        """Search for papers matching a text query and/or entity filters.

        Args:
            query: Natural language query text (will be embedded automatically).
                If None, performs entity-only search without vector ranking.
            top_k: Number of results to return.
            pmids: Optional list of PMIDs to restrict search to.
            entity_groups: CNF entity filter - outer list is AND, inner lists are OR.
                E.g. [["aspirin", "ibuprofen"], ["diabetes"]] means
                (aspirin OR ibuprofen) AND diabetes.
                Entity names will be normalized to IDs automatically.
                Negated terms (~entity) supported.
            scope: Filter scope.
            include_content: Whether to load full paper content from PaperStore.
            expand_entities: Whether to expand entities to include narrower/child concepts.
            negate_entity_filter: If True, invert the overall entity filter condition.
                Requires `entity_groups` and a vector query.
            stratify_by_pmid: Whether to ensure results are stratified by unique PMIDs.

        Returns:
            List of PaperSearchResult with metadata and optionally full content.

        Raises:
            ValueError: If neither query, entity_groups, nor pmids is provided.
            EntityNormalizationError: If any entity terms cannot be normalized.
        """
        ...

    @abstractmethod
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
        """Count papers/paragraphs matching entity filter and return a sample.

        Args:
            entity_groups: CNF entity filter - outer list is AND, inner lists are OR.
            sample_size: Number of sample results to return.
            scope: Filter scope - 'paper', 'chunk', or 'paragraph'.
            expand_entities: Whether to expand entities to include narrower/child concepts.
            include_content: Whether to load full paper content.
            use_expanded_bitmaps: Use precomputed expanded bitmaps when supported.
            expansion_depth: Depth for bitmap expansion when using precomputed tables.

        Returns:
            Tuple of (total_count, sample_results).

        Raises:
            EntityNormalizationError: If any entity terms cannot be normalized.
        """
        ...

    @abstractmethod
    async def score_pmids(
        self,
        query: str,
        pmids: Sequence[str],
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Score candidate pmids by vector similarity to an embedded query.

        Returns per-chunk `(pmid, similarity)` pairs for chunks belonging to
        `pmids`; higher is better. Multiple entries may be returned per pmid
        (one per matching chunk). The caller aggregates / reranks as needed
        — the ABC intentionally exposes only the raw similarity primitive.

        Args:
            query: Natural language query (will be embedded).
            pmids: Candidate pmids to restrict scoring to.
            limit: Maximum number of (pmid, score) rows to return.
        """
        ...

    @abstractmethod
    async def get_concept_info(self, concept_ids: Sequence[str]) -> dict[str, ConceptInfo]:
        """Return concept info for a list of global identifiers.

        Args:
            concept_ids: List of global identifier strings (e.g., "NCBITax:9606").

        Returns:
            Dict mapping global_identifier to ConceptInfo with preferred_name, etc.
        """
        ...

    @abstractmethod
    async def fetch_paragraph_entities(
        self,
        targets: Sequence[tuple[str, int]],
    ) -> list[dict[str, object]]:
        """Fetch distinct entity mentions for specific PMID + paragraph pairs.

        Args:
            targets: List of (pmid, paragraph_index) tuples to query.

        Returns:
            List of dicts with keys: pmid, paragraph, ent, global_identifier.
        """
        ...

    @abstractmethod
    async def fetch_paragraph_entity_mentions(
        self,
        targets: Sequence[tuple[str, int]],
    ) -> list[dict[str, object]]:
        """Fetch entity mentions (small molecules, genes, proteins) for specific PMID + paragraph pairs.

        Args:
            targets: List of (pmid, paragraph_index) tuples to query.

        Returns:
            List of dicts with keys:
                pmid, paragraph, name, alt, surface, global_identifier, entity_type.
        """
        ...

    @abstractmethod
    async def count_entity_mentions(
        self,
        entity_type: EntityType,
        min_mentions: int = 1,
        pmids: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Find papers with high mention counts for a given entity type.

        Args:
            entity_type: The entity type to count (e.g., "Gene", "Disease").
            min_mentions: Minimum number of mentions required.
            pmids: Optional list of PMIDs to restrict search to.
            limit: Maximum number of results to return.

        Returns:
            List of dicts with keys: pmid, total_mentions.
            Sorted by total_mentions descending.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (database connections, etc.)."""
        ...
