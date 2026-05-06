import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

import polars as pl
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, ModelRetry, RunContext
from pydantic_ai.models import Model
from tqdm.auto import tqdm

from starling.cli.conf.config import StarlingAgentTuningConfig
from starling.cli.extraction import extract_from_paper
from starling.cli.extraction_types import (
    ExtractionRecord,
    ExtractionSampleRecord,
    ExtractionSampleResult,
    ExtractionSampleState,
    ExtractorDeps,
)
from starling.cli.utils import handle_db_errors
from starling.infra.entities import EntityCNF, EntityType
from starling.infra.entity_filter import FilterScope
from starling.infra.milhouse_vector_search import MilhouseVectorSearch
from starling.infra.normalizer_base import EntityNormalizer
from starling.infra.papers import Paper
from starling.infra.vector_search_base import PaperSearchResult, VectorSearchBase
from starling.tools.normalize import normalize_entities_tool
from starling.utils.limited import limited_as_completed
from starling.utils.paper_expansion import expand_around_idx

TEMPLATE_DIR = files("starling") / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)


logger = logging.getLogger(__name__)


# --- Validator Agent ---


@dataclass
class ValidatorDeps:
    task: str
    criteria: str | None = None


class ValidationOutput(BaseModel):
    """Output from the validator agent for a single paper."""

    is_relevant: bool = Field(description="Whether the paper is relevant to the research task")
    reasoning: str = Field(description="Brief explanation of why the paper is or isn't relevant")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the relevance judgment (0-1)")


def build_validator_agent(model: Model) -> Agent[ValidatorDeps, ValidationOutput]:
    """Build an agent that validates whether a paper is relevant to a task."""

    def _get_validator_instructions(ctx: RunContext[ValidatorDeps]) -> str:
        template = jinja_env.get_template("starling/validator.jinja2")
        return template.render(
            task=ctx.deps.task,
            criteria=ctx.deps.criteria,
        )

    agent = Agent[ValidatorDeps, ValidationOutput](
        model=model,
        output_type=ValidationOutput,
        deps_type=ValidatorDeps,
        instructions=_get_validator_instructions,
        retries=4,
    )
    return agent


# --- Investigator Agent ---

InvestigationFocus = Literal["false_positive", "missed_relevant", "terminology", "extraction_failure"]


@dataclass
class InvestigatorDeps:
    task: str
    focus: InvestigationFocus
    current_filter: str | None
    paper_entities: list[str] | None = None


class PaperInsight(BaseModel):
    """Deep analysis of a paper for filter refinement."""

    relevance_assessment: str = Field(description="Why this paper is or isn't relevant to the task")
    key_terminology: list[str] = Field(
        default_factory=list,
        description="Terms/phrases used in this paper that could improve the filter (e.g., 'brain uptake', 'Kp,brain', 'CNS penetration')",
    )
    entity_types_observed: list[str] = Field(
        default_factory=list,
        description="Entity types mentioned that might be missing from the filter (e.g., 'Assay/Result', 'Pathway')",
    )
    suggested_additions: list[str] = Field(
        default_factory=list,
        description="Specific terms to ADD to the filter to catch papers like this",
    )
    suggested_exclusions: list[str] = Field(
        default_factory=list,
        description="Terms that would help EXCLUDE false positives like this",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the analysis (0-1)")


def build_investigator_agent(model: Model) -> Agent[InvestigatorDeps, PaperInsight]:
    """Build an agent that deeply analyzes papers for filter refinement insights."""

    def _get_investigator_instructions(ctx: RunContext[InvestigatorDeps]) -> str:
        template = jinja_env.get_template("starling/investigator.jinja2")
        return template.render(
            task=ctx.deps.task,
            focus=ctx.deps.focus,
            current_filter=ctx.deps.current_filter,
            paper_entities=ctx.deps.paper_entities,
        )

    agent = Agent[InvestigatorDeps, PaperInsight](
        model=model,
        output_type=PaperInsight,
        deps_type=InvestigatorDeps,
        instructions=_get_investigator_instructions,
        retries=4,
    )
    return agent


# --- Corpus Filter Agent ---


@dataclass
class CorpusFilterDeps:
    task: str
    validator: Agent[ValidatorDeps, ValidationOutput]
    investigator: Agent[InvestigatorDeps, PaperInsight]
    searcher: VectorSearchBase
    tuning: StarlingAgentTuningConfig
    # Optional extraction sampling (if None, sample_extraction tool is disabled)
    extractor: "Agent[ExtractorDeps, list[ExtractionRecord]] | None" = None
    extraction_state: ExtractionSampleState | None = None


class PaperHit(BaseModel):
    """A paper result from search (entity or semantic)."""

    pmid: str
    title: str | None
    snippet: str | None = Field(default=None, description="Longest matching paragraph text")
    snippet_entities: list[str] | None = Field(
        default=None,
        description="Entity names mentioned in the snippet",
    )
    score: float | None = Field(default=None, description="Similarity score (for semantic search)")


class EntitySearchResult(BaseModel):
    count: int
    sample: list[PaperHit]


class EntityCount(BaseModel):
    pmid: str
    total_mentions: int


class ValidatedPaper(BaseModel):
    """A paper with its validation judgment."""

    pmid: str
    title: str | None
    is_relevant: bool
    reasoning: str
    confidence: float


class FilterValidationResult(BaseModel):
    """Result of validating a filter by sampling and judging papers."""

    filter_yield: int = Field(
        description="Total papers matching the filter; for semantic queries, equals the sampled result count"
    )
    sample_size: int = Field(description="Number of papers sampled for validation")
    relevant_count: int = Field(description="Number of sampled papers judged relevant")
    precision_estimate: float = Field(description="Estimated precision (relevant_count / sample_size)")
    examples: list[ValidatedPaper] = Field(description="Validated papers with judgments")


class RecallProbeResult(BaseModel):
    """Result of probing for missed papers outside the current filter."""

    probe_type: Literal["random", "semantic"] = Field(description="Type of recall probe used")
    filter_yield: int = Field(description="Papers matching current filter")
    outside_sample_size: int = Field(description="Papers sampled outside filter")
    relevant_outside: int = Field(description="Outside samples judged relevant")
    estimated_recall_gap: float = Field(description="Fraction of outside sample that was relevant")
    examples: list[ValidatedPaper] = Field(description="Validated outside papers")


class MarginalRecallGainResult(BaseModel):
    """Estimated value of adding one candidate sub-filter to an existing filter set."""

    base_pool_size: int = Field(description="Total unique PMIDs already covered by accepted filters")
    candidate_pool_size: int = Field(description="PMIDs returned by the candidate filter")
    novelty_pct: float = Field(description="% of candidate results not already in the base pool (0-100)")
    novel_precision_pct: float = Field(description="% of novel papers judged relevant (0-100)")
    accept: bool = Field(description="Whether this filter passes the acceptance threshold")
    examples: list[ValidatedPaper] = Field(description="Sample of validated novel papers")


class SemanticSearchResult(BaseModel):
    """Result from semantic search."""

    query: str = Field(description="The semantic query used")
    hits: list[PaperHit] = Field(description="Papers matching the query")


@dataclass
class ValidationCandidate:
    pmid: str
    title: str | None
    abstract: str | None
    snippet: str | None = None


def _select_snippet(paper: Paper | None, indices: list[int]) -> tuple[str | None, int | None]:
    if paper is None or not indices:
        return None, None

    idx_set = set(indices)
    matches = [(ts.idx, text) for ts in paper.flat_tagged_strings() if ts.idx in idx_set and (text := ts.text.strip())]
    if not matches:
        return None, None

    snippet_idx, snippet_text = max(matches, key=lambda item: len(item[1]))
    return snippet_text, snippet_idx


def _candidate_snippet_indices(result: PaperSearchResult) -> list[int]:
    if result.matched_paragraphs and result.chunks:
        overlap = set(result.matched_paragraphs) & set(result.chunks)
        return list(overlap) if overlap else list(result.chunks)
    if result.matched_paragraphs:
        return result.matched_paragraphs[:5]
    return list(result.chunks)


async def _fetch_snippet_entities(
    searcher: VectorSearchBase,
    targets: list[tuple[str, int]],
    max_entities: int,
) -> dict[tuple[str, int], list[str]]:
    if not targets:
        return {}

    rows = await searcher.fetch_paragraph_entities(targets)
    if not rows:
        return {}

    global_ids = {row["global_identifier"] for row in rows if row.get("global_identifier")}
    concept_info = await searcher.get_concept_info(list(global_ids)) if global_ids else {}  # ty:ignore[invalid-argument-type]

    entities_by_target: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        pmid = str(row["pmid"])
        paragraph = int(row["paragraph"])  # ty:ignore[invalid-argument-type]
        gid = row["global_identifier"]
        if gid is None:
            continue

        info = concept_info.get(gid)  # ty:ignore[invalid-argument-type]
        label = info.preferred_name if info is not None else str(gid)
        entities_by_target.setdefault((pmid, paragraph), {})[str(gid)] = label

    return {
        key: sorted(entities.values())[:max_entities] if max_entities else sorted(entities.values())
        for key, entities in entities_by_target.items()
    }


async def _build_paper_hits(
    searcher: VectorSearchBase,
    results: list[PaperSearchResult],
    *,
    include_score: bool,
    max_snippet_entities: int,
) -> list[PaperHit]:
    rows: list[tuple[PaperSearchResult, str | None, int | None]] = []
    snippet_targets: list[tuple[str, int]] = []
    for result in results:
        snippet, snippet_idx = _select_snippet(result.content, _candidate_snippet_indices(result))
        if snippet_idx is not None:
            snippet_targets.append((str(result.pmid), snippet_idx))
        rows.append((result, snippet, snippet_idx))

    snippet_entities = await _fetch_snippet_entities(searcher, snippet_targets, max_entities=max_snippet_entities)

    hits = []
    for result, snippet, snippet_idx in rows:
        entities = None
        if snippet_idx is not None:
            entities = snippet_entities.get((str(result.pmid), snippet_idx))

        hits.append(
            PaperHit(
                pmid=str(result.pmid),
                title=result.title,
                snippet=snippet,
                snippet_entities=entities,
                score=result.vec_sim if include_score else None,
            )
        )

    return hits


async def _validate_candidates(
    ctx: RunContext[CorpusFilterDeps],
    candidates: list[ValidationCandidate],
    relevance_criteria: str | None = None,
) -> tuple[list[ValidatedPaper], int]:

    def _build_validation_prompt(candidate: ValidationCandidate) -> str:
        prompt = f"Title: {candidate.title or 'N/A'}"
        if candidate.snippet is not None:
            prompt += f"\n\nSnippet: {candidate.snippet}"
        return prompt

    if not candidates:
        return [], 0

    task_deps = ValidatorDeps(
        task=ctx.deps.task,
        criteria=relevance_criteria,
    )
    validation_tasks = [
        ctx.deps.validator.run(_build_validation_prompt(candidate), deps=task_deps) for candidate in candidates
    ]
    validation_results: list[AgentRunResult[ValidationOutput]] = await asyncio.gather(*validation_tasks)

    validated_papers: list[ValidatedPaper] = []
    relevant_count = 0

    for candidate, result in zip(candidates, validation_results):
        output = result.output
        if output.is_relevant:
            relevant_count += 1

        validated_papers.append(
            ValidatedPaper(
                pmid=candidate.pmid,
                title=candidate.title,
                is_relevant=output.is_relevant,
                reasoning=output.reasoning,
                confidence=output.confidence,
            )
        )

    return validated_papers, relevant_count


# --- Unified Output Type ---


class FilterSpec(BaseModel):
    """A single filter: entity groups + semantic query."""

    entity_groups: EntityCNF = Field(description="CNF entity filter (outer=AND, inner=OR)")
    expand_entities: bool = Field(default=True, description="Whether to expand to child/narrower concepts")
    semantic_query: str = Field(description="Natural language query for semantic search")

    marginal_recall_gain: float | None = Field(
        default=None,
        description="Estimated marginal recall gain when this filter was proposed",
    )
    estimated_count: int | None = Field(default=None, description="Estimated number of papers matching")
    precision_estimate: float | None = Field(default=None, description="Estimated precision from validation")


class RetrievalSpec(BaseModel):
    """Agent output: a list of high-precision filters whose union forms the retrieval set."""

    filters: list[FilterSpec] = Field(description="Union of high-precision filters")


def build_corpus_filter_agent(
    model: Model,
    normalizer: EntityNormalizer,
    *,
    include_extraction: bool = False,
) -> Agent[CorpusFilterDeps, RetrievalSpec]:

    def _get_instructions(ctx: RunContext[CorpusFilterDeps]) -> str:
        template = jinja_env.get_template("starling/starling_single.jinja2")
        return template.render(
            task=ctx.deps.task,
            target_precision=ctx.deps.tuning.target_precision,
            target_recall_gap=ctx.deps.tuning.target_recall_gap,
            target_extraction_rate=ctx.deps.tuning.target_extraction_rate,
            marginal_recall_gain_threshold=ctx.deps.tuning.marginal_recall_gain_threshold,
            max_sub_filters=ctx.deps.tuning.max_sub_filters,
        )

    tools = [
        normalize_entities_tool(normalizer, allow_partial=False),
        count_entity_mentions,
        search_papers_by_entities,
        validate_filter,
        semantic_search_papers,
        probe_recall_semantic,
        estimate_marginal_recall_gain,
        investigate_paper,
    ]

    # Only include sample_extraction if extraction is enabled
    if include_extraction:
        tools.append(sample_extraction)  # ty:ignore[invalid-argument-type]

    agent = Agent[CorpusFilterDeps, RetrievalSpec](
        model=model,
        output_type=RetrievalSpec,
        deps_type=CorpusFilterDeps,
        retries=4,
        instructions=_get_instructions,
        tools=tools,
    )

    return agent


@handle_db_errors
async def count_entity_mentions(
    ctx: RunContext[CorpusFilterDeps],
    entity_type: EntityType,
    min_mentions: int = 1,
    pmids: list[str] | None = None,
    limit: int = 100,
) -> list[EntityCount]:
    """Find papers with high entity mention counts.

    Args:
        ctx: The run context with dependencies.
        entity_type: The type of entity to count (e.g., 'SmallMolecule', 'Disease').
        min_mentions: Minimum total mentions required.
        pmids: Optional list of PMIDs to filter (for chaining with other searches).
        limit: Maximum number of results to return.

    Returns:
        List of (PMID, total_mentions) sorted by mention count descending.
    """
    if pmids is not None and len(pmids) == 0:
        return []

    rows = await ctx.deps.searcher.count_entity_mentions(
        entity_type,
        min_mentions=min_mentions,
        pmids=pmids,
        limit=limit,
    )
    return [
        EntityCount(
            pmid=str(row["pmid"]),
            total_mentions=int(row["total_mentions"]),  # ty:ignore[invalid-argument-type]
        )
        for row in rows
    ]


@handle_db_errors
async def search_papers_by_entities(
    ctx: RunContext[CorpusFilterDeps],
    entity_groups: EntityCNF,
    expand_entities: bool = True,
    sample_size: int = 10,
) -> EntitySearchResult:
    """Find papers containing specific entities using CNF (AND/OR) logic.

    Provide natural language entity terms - they will be automatically normalized.

    Entity groups use Conjunctive Normal Form:
    - Outer list: groups that must ALL match (AND)
    - Inner list: alternatives within a group where ANY can match (OR)

    Examples:
    - [["aspirin"]] - papers mentioning aspirin
    - [["aspirin"], ["heart disease"]] - papers mentioning aspirin AND heart disease
    - [["aspirin", "ibuprofen"]] - papers mentioning aspirin OR ibuprofen
    - [["breast cancer"], ["BRCA1", "BRCA2"]] - breast cancer AND (BRCA1 OR BRCA2)

    Args:
        ctx: The run context with dependencies.
        entity_groups: CNF filter using natural language entity terms.
        expand_entities: If true, also match child/narrower concepts (e.g., "cancer" would also match "breast cancer").
        sample_size: Number of sample PMIDs to return (default 10).

    Returns:
        Count of matching papers and a sample of PMIDs.
    """
    if not entity_groups or not any(entity_groups):
        return EntitySearchResult(count=0, sample=[])

    count, results = await ctx.deps.searcher.count_and_sample(
        entity_groups,
        sample_size=sample_size,
        expand_entities=expand_entities,
        scope=ctx.deps.tuning.filter_scope,
        include_content=True,
    )

    sample = await _build_paper_hits(
        ctx.deps.searcher,
        results,
        include_score=False,
        max_snippet_entities=ctx.deps.tuning.max_snippet_entities,
    )

    return EntitySearchResult(count=count, sample=sample)


@handle_db_errors
async def validate_filter(
    ctx: RunContext[CorpusFilterDeps],
    entity_groups: EntityCNF,
    semantic_query: str,
    expand_entities: bool = True,
    sample_size: int = 100,
    relevance_criteria: str | None = None,
) -> FilterValidationResult:
    """Sample papers from a filter and validate their relevance to the task.

    Requires both entity_groups (for precise filtering) and semantic_query
    (for relevance ranking). Returns precision estimate and example judgments.

    Args:
        ctx: The run context with dependencies.
        entity_groups: CNF filter using natural language entity terms (same as search_papers_by_entities).
        semantic_query: Natural language query for semantic validation.
        expand_entities: If true, also match child/narrower concepts.
        sample_size: Number of papers to sample and validate (default 100).
        relevance_criteria: Optional refined definition of what makes a paper relevant.
            Use this to guide the validator after you've learned what distinguishes
            true positives from false positives.

    Returns:
        Validation result with precision estimate and example judgments. A small set of examples is returned for context, but the precision estimate is based on the full sample.
    """
    # Run count (for true filter yield) and semantic search (for ranked samples) in parallel
    count_coro = ctx.deps.searcher.count_and_sample(
        entity_groups,
        sample_size=0,
        expand_entities=expand_entities,
        scope=ctx.deps.tuning.filter_scope,
        include_content=False,
    )
    search_coro = ctx.deps.searcher.search(
        semantic_query.strip(),
        top_k=sample_size,
        entity_groups=entity_groups,
        expand_entities=expand_entities,
        scope=ctx.deps.tuning.filter_scope,
        include_content=True,
        stratify_by_pmid=True,
    )

    (filter_yield, _), results = await asyncio.gather(count_coro, search_coro)

    if not results:
        return FilterValidationResult(
            filter_yield=filter_yield,
            sample_size=0,
            relevant_count=0,
            precision_estimate=0.0,
            examples=[],
        )

    # Dedupe over PMIDs
    candidates = []
    seen_pmids = set()
    for r in results:
        if r.pmid in seen_pmids:
            continue

        seen_pmids.add(r.pmid)

        # For validation, show full (or large part of) paper
        snippet = None
        snippet_indices = _candidate_snippet_indices(r)
        if r.content is not None and snippet_indices:
            center = snippet_indices[len(snippet_indices) // 2]
            if center is not None:
                snippet = expand_around_idx(
                    paper=r.content,
                    center_idx=center,
                    total_tokens=ctx.deps.tuning.validation_snippet_tokens,
                ).to_markdown()

        candidates.append(
            ValidationCandidate(
                pmid=r.pmid,
                title=r.title,
                abstract=r.abstract,
                snippet=snippet,
            )
        )
    validated_papers, relevant_count = await _validate_candidates(
        ctx,
        candidates,
        relevance_criteria=relevance_criteria,
    )

    precision = relevant_count / len(validated_papers) if validated_papers else 0.0

    max_examples = max(0, ctx.deps.tuning.max_examples_returned)
    # Choose a set of relevant and irrelevant examples to return (up to max_examples)
    relevant_examples = [p for p in validated_papers if p.is_relevant][: max_examples // 2]
    irrelevant_examples = [p for p in validated_papers if not p.is_relevant][: max_examples // 2]
    examples = relevant_examples + irrelevant_examples
    if len(examples) < max_examples:
        remaining = max_examples - len(examples)
        additional_examples = [p for p in validated_papers if p not in examples][:remaining]
        examples.extend(additional_examples)

    return FilterValidationResult(
        filter_yield=filter_yield,
        sample_size=len(validated_papers),
        relevant_count=relevant_count,
        precision_estimate=precision,
        examples=examples,
    )


@handle_db_errors
async def semantic_search_papers(
    ctx: RunContext[CorpusFilterDeps],
    query: str,
    entity_groups: EntityCNF | None = None,
    expand_entities: bool = True,
    top_k: int = 20,
) -> SemanticSearchResult:
    """Search for papers by semantic similarity to a natural language query.

    Use this when exploring RELATIONSHIPS between entities that entity filters cannot capture.
    For example: "this molecule crosses the blood-brain barrier" or "drug X inhibits protein Y".

    Args:
        ctx: The run context with dependencies.
        query: Natural language query describing what you're looking for.
        entity_groups: Optional CNF entity filter to pre-filter papers BEFORE semantic search.
        expand_entities: Whether to expand child entities in the filter (default True).
        top_k: Number of results to return (default 20).

    Returns:
        Papers with titles, snippets, and similarity scores.
    """
    results = await ctx.deps.searcher.search(
        query,
        top_k=top_k,
        entity_groups=entity_groups or None,
        expand_entities=expand_entities,
        scope=ctx.deps.tuning.filter_scope,
        include_content=True,
    )

    hits = await _build_paper_hits(
        ctx.deps.searcher,
        results,
        include_score=True,
        max_snippet_entities=ctx.deps.tuning.max_snippet_entities,
    )

    return SemanticSearchResult(query=query, hits=hits)


# --- Recall Probing Tools ---


@handle_db_errors
async def probe_recall_semantic(
    ctx: RunContext[CorpusFilterDeps],
    entity_groups: EntityCNF,
    semantic_query: str,
    expand_entities: bool = True,
    boundary_k: int = 50,
    sample_size: int = 50,
) -> RecallProbeResult:
    """Find papers semantically similar to your query but NOT matching your entity filter.

    This shows the 'boundary' of your filter - papers that LOOK relevant (semantically similar)
    but don't match your entity criteria. High relevance suggests your filter is too narrow.

    Args:
        ctx: The run context with dependencies.
        entity_groups: CNF filter to probe (same format as search_papers_by_entities).
        semantic_query: Natural language query describing relevant papers.
        expand_entities: Whether to expand to child/narrower concepts.
        boundary_k: Number of semantic results to fetch for the outside pool (default 200).
        sample_size: Number of outside papers to validate (default 100).

    Returns:
        RecallProbeResult with relevance estimate for semantically similar but unmatched papers.
    """
    if not entity_groups or not any(entity_groups):
        raise ModelRetry("entity_groups cannot be empty for recall probing")

    (filter_yield, _), semantic_results = await asyncio.gather(
        ctx.deps.searcher.count_and_sample(
            entity_groups,
            sample_size=0,
            scope=ctx.deps.tuning.filter_scope,
            expand_entities=expand_entities,
        ),
        ctx.deps.searcher.search(
            semantic_query,
            top_k=boundary_k,
            entity_groups=entity_groups,
            negate_entity_filter=True,
            expand_entities=expand_entities,
            scope=ctx.deps.tuning.filter_scope,
            include_content=True,
            stratify_by_pmid=True,
        ),
    )

    boundary_papers: list[PaperSearchResult] = []
    seen_pmids: set[str] = set()
    for result in semantic_results:
        if result.pmid not in seen_pmids:
            seen_pmids.add(result.pmid)
            boundary_papers.append(result)
            if len(boundary_papers) >= sample_size:
                break

    if not boundary_papers:
        return RecallProbeResult(
            probe_type="semantic",
            filter_yield=filter_yield,
            outside_sample_size=0,
            relevant_outside=0,
            estimated_recall_gap=0.0,
            examples=[],
        )

    candidates = []
    for result in boundary_papers:
        snippet, _ = _select_snippet(result.content, _candidate_snippet_indices(result))
        candidates.append(
            ValidationCandidate(
                pmid=str(result.pmid),
                title=result.title,
                abstract=result.abstract,
                snippet=snippet,
            )
        )

    validated_papers, relevant_count = await _validate_candidates(ctx, candidates)

    recall_gap = relevant_count / len(validated_papers) if validated_papers else 0.0

    return RecallProbeResult(
        probe_type="semantic",
        filter_yield=filter_yield,
        outside_sample_size=len(validated_papers),
        relevant_outside=relevant_count,
        estimated_recall_gap=recall_gap,
        examples=validated_papers[: max(0, ctx.deps.tuning.max_examples_returned)],
    )


@handle_db_errors
async def estimate_marginal_recall_gain(
    ctx: RunContext[CorpusFilterDeps],
    accepted_filters: list[FilterSpec],
    candidate_filter: FilterSpec,
    relevance_criteria: str | None = None,
    validation_sample_size: int | None = None,
) -> MarginalRecallGainResult:
    """Estimate marginal recall gain from adding candidate_filter to accepted_filters.

    Validates a sample of candidate PMIDs that are novel relative to the current
    accepted filter union and reports novelty %, novel precision %, and an
    accept/reject decision.
    """
    max_examples = max(0, ctx.deps.tuning.max_examples_returned)
    gain_threshold = ctx.deps.tuning.marginal_recall_gain_threshold

    base_k = max(1, ctx.deps.tuning.marginal_gain_base_k)
    candidate_k = max(1, ctx.deps.tuning.marginal_gain_candidate_k)

    validation_sample_size = max(1, validation_sample_size or ctx.deps.tuning.marginal_gain_validation_sample_size)

    # Build base coverage pool from accepted filters.
    base_seen: set[str] = set()
    for sub_filter in accepted_filters:
        sub_results = await _execute_single_filter(
            searcher=ctx.deps.searcher,
            entity_groups=sub_filter.entity_groups,
            semantic_query=sub_filter.semantic_query,
            expand_entities=sub_filter.expand_entities,
            scope=ctx.deps.tuning.filter_scope,
            limit=base_k,
        )
        base_seen.update(r.pmid for r in sub_results)

    candidate_results = await _execute_single_filter(
        searcher=ctx.deps.searcher,
        entity_groups=candidate_filter.entity_groups,
        semantic_query=candidate_filter.semantic_query,
        expand_entities=candidate_filter.expand_entities,
        scope=ctx.deps.tuning.filter_scope,
        limit=candidate_k,
    )
    candidate_pmids = [r.pmid for r in candidate_results]

    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return max(0.0, min(1.0, numerator / denominator))

    def _acceptance_score(*, novel_precision: float, marginal_gain: float) -> float:
        return marginal_gain if base_seen else novel_precision

    base_validation_pmids = list(base_seen)[:validation_sample_size]
    estimated_base_relevant = 0.0
    if base_validation_pmids:
        base_validated_papers, base_relevant = await _fetch_and_validate(
            ctx,
            base_validation_pmids,
            relevance_criteria=relevance_criteria,
        )
        base_precision = base_relevant / len(base_validated_papers) if base_validated_papers else 0.0
        estimated_base_relevant = base_precision * len(base_seen)

    novel_pmids = [pmid for pmid in candidate_pmids if pmid not in base_seen]
    if not novel_pmids:
        score = _acceptance_score(novel_precision=0.0, marginal_gain=_safe_ratio(0.0, estimated_base_relevant))
        return MarginalRecallGainResult(
            base_pool_size=len(base_seen),
            candidate_pool_size=len(candidate_pmids),
            novelty_pct=0.0,
            novel_precision_pct=0.0,
            accept=score >= gain_threshold,
            examples=[],
        )

    validation_pmids = novel_pmids[:validation_sample_size]
    validated_papers, relevant_count = await _fetch_and_validate(
        ctx,
        validation_pmids,
        relevance_criteria=relevance_criteria,
    )
    novel_precision = relevant_count / len(validated_papers) if validated_papers else 0.0
    estimated_new_relevant = novel_precision * len(novel_pmids)
    marginal_gain = _safe_ratio(estimated_new_relevant, estimated_base_relevant + estimated_new_relevant)
    novelty_pct = 100.0 * len(novel_pmids) / len(candidate_pmids) if candidate_pmids else 0.0
    score = _acceptance_score(novel_precision=novel_precision, marginal_gain=marginal_gain)

    return MarginalRecallGainResult(
        base_pool_size=len(base_seen),
        candidate_pool_size=len(candidate_pmids),
        novelty_pct=novelty_pct,
        novel_precision_pct=100.0 * novel_precision,
        accept=score >= gain_threshold,
        examples=validated_papers[:max_examples],
    )


# --- Extraction Sampling Tool ---


@handle_db_errors
async def sample_extraction(
    ctx: RunContext[CorpusFilterDeps],
    entity_groups: EntityCNF,
    semantic_query: str,
    expand_entities: bool = True,
    sample_size: int | None = None,
) -> ExtractionSampleResult:
    """Sample extraction from papers matching the current filter.

    Call this after validate_filter passes to verify that relevant papers
    actually contain extractable structured data. This catches "relevant but
    not extractable" failures early.

    Requires both entity_groups (for precise filtering) and semantic_query
    (for relevance ranking).

    The tool tracks cumulative statistics across calls, measuring:
    - Extraction rate (what fraction of papers yield extractions)
    - Unique entity discovery (are we still finding new things?)
    - Marginal yield (is discovery declining - a convergence signal)

    Args:
        ctx: The run context with dependencies.
        entity_groups: CNF entity filter (same format as search_papers_by_entities).
        semantic_query: Semantic search query.
        expand_entities: Whether to expand to child/narrower concepts.
        sample_size: Number of papers to attempt extraction on (defaults to tuning.extraction_sample_default_size).

    Returns:
        ExtractionSampleResult with extraction metrics and sample records.
    """
    # Check if extraction is enabled
    if ctx.deps.extractor is None or ctx.deps.extraction_state is None:
        raise ModelRetry(
            "Extraction sampling is not enabled. "
            "The extractor and extraction_state must be provided in CorpusFilterDeps."
        )

    if sample_size is None:
        sample_size = ctx.deps.tuning.extraction_sample_default_size
    sample_size = max(0, sample_size)
    if sample_size == 0:
        return ExtractionSampleResult(
            papers_attempted=0,
            papers_with_extractions=0,
            papers_failed=0,
            extraction_rate=0.0,
            total_extractions=0,
            avg_extractions_per_paper=0.0,
            sample_records=[],
            failure_reasons={},
            cumulative_papers_sampled=ctx.deps.extraction_state.total_papers_sampled,
            cumulative_extractions=ctx.deps.extraction_state.total_extractions,
            cumulative_extraction_rate=ctx.deps.extraction_state.overall_extraction_rate,
            cumulative_unique_entities=ctx.deps.extraction_state.total_unique_identifiers,
            new_unique_this_sample=0,
            marginal_yield_declining=ctx.deps.extraction_state.marginal_yield_declining,
        )

    # Get sample PMIDs via semantic search with entity pre-filter
    results = await ctx.deps.searcher.search(
        semantic_query.strip(),
        top_k=sample_size,
        entity_groups=entity_groups,
        expand_entities=expand_entities,
        scope=ctx.deps.tuning.filter_scope,
        include_content=False,
        stratify_by_pmid=True,
    )
    pmids = list(dict.fromkeys(r.pmid for r in results))[:sample_size]

    if not pmids:
        return ExtractionSampleResult(
            papers_attempted=0,
            papers_with_extractions=0,
            papers_failed=0,
            extraction_rate=0.0,
            total_extractions=0,
            avg_extractions_per_paper=0.0,
            sample_records=[],
            failure_reasons={},
            cumulative_papers_sampled=ctx.deps.extraction_state.total_papers_sampled,
            cumulative_extractions=ctx.deps.extraction_state.total_extractions,
            cumulative_extraction_rate=ctx.deps.extraction_state.overall_extraction_rate,
            cumulative_unique_entities=ctx.deps.extraction_state.total_unique_identifiers,
            new_unique_this_sample=0,
            marginal_yield_declining=ctx.deps.extraction_state.marginal_yield_declining,
        )

    # Run extraction on each paper
    papers_with_extractions = 0
    papers_failed = 0
    all_extractions: list = []
    failure_reasons: dict[str, int] = {}

    extraction_tasks = [
        lambda pmid=pmid: extract_from_paper(
            pmid=pmid,
            task=ctx.deps.task,
            extractor=ctx.deps.extractor,  # ty:ignore[invalid-argument-type]
            paper_store=ctx.deps.searcher.paper_store,
            searcher=ctx.deps.searcher,
            tokens_per_paper=ctx.deps.tuning.extraction_sample_tokens_per_paper,
        )
        for pmid in pmids
    ]
    parallelism = min(ctx.deps.tuning.extraction_sample_parallelism, len(pmids))
    async for result in limited_as_completed(
        extraction_tasks,
        in_flight=parallelism,
        return_exceptions=True,
        show_progress=False,
    ):
        if isinstance(result, BaseException):
            papers_failed += 1
            reason = type(result).__name__
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue

        if result.status == "success" and result.extractions:
            papers_with_extractions += 1
            all_extractions.extend(result.extractions)
        elif result.status == "failed":
            papers_failed += 1
            reason = result.error or "unknown"
            # Simplify error messages
            if ":" in reason:
                reason = reason.split(":")[0]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        else:
            # no_content or success with no extractions
            reason = "no_extractable_content"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    # Collect global identifiers for entity deduplication
    global_identifiers = [r.global_identifier for r in all_extractions if r.global_identifier]

    # Update cumulative state
    effective_attempts = max(0, len(pmids) - papers_failed)
    new_unique = ctx.deps.extraction_state.add_sample(
        papers_attempted=effective_attempts,
        papers_with_extractions=papers_with_extractions,
        num_extractions=len(all_extractions),
        global_identifiers=global_identifiers,
    )

    # Build sample records for agent review (limit to 10)
    sample_records: list[ExtractionSampleRecord] = []
    for record in all_extractions[:10]:
        sample_records.append(
            ExtractionSampleRecord(
                pmid=record.pmid,
                extraction_id=record.extraction_id,
                paragraph_idx=record.support.paragraph_idx,
                support_text=record.support.support_text[:500],  # Truncate long text
                extracted=record.extracted,
                global_identifier=record.global_identifier,
                confidence=record.confidence,
                needs_more_context=record.needs_more_context,
            )
        )

    extraction_rate = papers_with_extractions / effective_attempts if effective_attempts > 0 else 0.0
    avg_per_paper = len(all_extractions) / papers_with_extractions if papers_with_extractions > 0 else 0.0

    return ExtractionSampleResult(
        papers_attempted=len(pmids),
        papers_with_extractions=papers_with_extractions,
        papers_failed=papers_failed,
        extraction_rate=extraction_rate,
        total_extractions=len(all_extractions),
        avg_extractions_per_paper=avg_per_paper,
        sample_records=sample_records,
        failure_reasons=failure_reasons,
        cumulative_papers_sampled=ctx.deps.extraction_state.total_papers_sampled,
        cumulative_extractions=ctx.deps.extraction_state.total_extractions,
        cumulative_extraction_rate=ctx.deps.extraction_state.overall_extraction_rate,
        cumulative_unique_entities=ctx.deps.extraction_state.total_unique_identifiers,
        new_unique_this_sample=new_unique,
        marginal_yield_declining=ctx.deps.extraction_state.marginal_yield_declining,
    )


# --- Paper Investigation Tool ---


async def _fetch_paper_entities(
    searcher: VectorSearchBase,
    pmid: str,
    max_entities: int,
) -> list[str]:
    """Fetch all unique entities for a paper, deduped by global_identifier, returning preferred names."""
    # Fetch all paragraph indices for this paper - we use a broad range
    # The fetch_paragraph_entities expects (pmid, paragraph) tuples
    # We'll fetch for paragraphs 0-999 which should cover any paper
    targets = [(pmid, i) for i in range(1000)]

    rows = await searcher.fetch_paragraph_entities(targets)
    if not rows:
        return []

    # Collect unique global identifiers
    global_ids: set[str] = {str(row["global_identifier"]) for row in rows if row.get("global_identifier")}
    if not global_ids:
        return []

    # Get preferred names for all entities
    concept_info = await searcher.get_concept_info(list(global_ids))

    # Build deduped list with preferred names (global_ids is a set, so already unique)
    entities = {
        gid: (info.preferred_name if (info := concept_info.get(gid)) and info.preferred_name else gid)
        for gid in global_ids
    }

    sorted_entities = sorted(entities.values())
    return sorted_entities[:max_entities] if max_entities else sorted_entities


@handle_db_errors
async def investigate_paper(
    ctx: RunContext[CorpusFilterDeps],
    pmid: str,
    focus: InvestigationFocus = "terminology",
    current_filter_description: str | None = None,
) -> PaperInsight:
    """Deeply investigate a paper to extract insights for filter refinement.

    Use this when you need to understand:
    - WHY a paper matched your filter but wasn't relevant (focus="false_positive")
    - WHY a relevant paper was missed by your filter (focus="missed_relevant")
    - What terminology/entities are used in relevant papers (focus="terminology")
    - WHY extraction failed or yielded no data (focus="extraction_failure")

    This is powerful for discovering new filter terms and understanding failure patterns.
    Use sparingly (1-3 papers) after validation or recall probing reveals issues.

    Args:
        ctx: The run context with dependencies.
        pmid: The PMID of the paper to investigate.
        focus: What to focus the investigation on:
            - "false_positive": Understand why this irrelevant paper matched
            - "missed_relevant": Understand why this relevant paper was missed
            - "terminology": Extract useful terminology from a relevant paper
            - "extraction_failure": Understand why extraction failed or yielded no data
        current_filter_description: Optional description of your current filter for context.

    Returns:
        PaperInsight with terminology, suggested additions/exclusions, and analysis.
    """
    # Load the paper
    paper = await ctx.deps.searcher.paper_store.aget(pmid)
    if paper is None:
        raise ModelRetry(f"Paper {pmid} not found in corpus. Try a different PMID.")

    # Get a reasonable window of the paper content
    # Start from the middle of the paper for broad coverage
    all_strings = paper.flat_tagged_strings()
    if not all_strings:
        raise ModelRetry(f"Paper {pmid} has no content.")

    # Use the middle paragraph as center for balanced coverage
    center_idx = all_strings[len(all_strings) // 2].idx
    paper_window = expand_around_idx(
        paper,
        center_idx=center_idx,
        total_tokens=ctx.deps.tuning.investigation_max_tokens,
        drop_empty_sections=True,
    )

    # Format paper content for the investigator
    paper_content = paper_window.to_markdown(only_body=True)

    # Fetch entities for this paper to help the investigator understand what's indexed
    paper_entities = await _fetch_paper_entities(
        ctx.deps.searcher,
        pmid,
        max_entities=ctx.deps.tuning.investigation_max_entities,
    )

    # Run the investigator agent
    investigator_deps = InvestigatorDeps(
        task=ctx.deps.task,
        focus=focus,
        current_filter=current_filter_description,
        paper_entities=paper_entities or None,
    )

    result = await ctx.deps.investigator.run(
        f"Paper PMID: {pmid}\n\n{paper_content}",
        deps=investigator_deps,
    )

    return result.output


async def _fetch_and_validate(
    ctx: RunContext[CorpusFilterDeps],
    pmids: list[str],
    *,
    relevance_criteria: str | None = None,
) -> tuple[list[ValidatedPaper], int]:
    """Fetch metadata for PMIDs and validate relevance. Returns (validated_papers, relevant_count)."""
    rows = await ctx.deps.searcher.search(
        query=None,
        top_k=len(pmids),
        pmids=pmids,
        scope=ctx.deps.tuning.filter_scope,
        include_content=False,
    )
    by_pmid = {row.pmid: row for row in rows}
    candidates = []
    for pmid in pmids:
        row = by_pmid.get(pmid)
        candidates.append(
            ValidationCandidate(
                pmid=pmid,
                title=row.title if row else None,
                abstract=row.abstract if row else None,
                snippet=None,
            )
        )
    return await _validate_candidates(
        ctx,
        candidates,
        relevance_criteria=relevance_criteria,
    )


async def _execute_single_filter(
    *,
    searcher: VectorSearchBase,
    entity_groups: EntityCNF,
    semantic_query: str,
    expand_entities: bool,
    scope: FilterScope,
    limit: int,
) -> list[PaperSearchResult]:
    """Execute one filter definition and return deduped results (one per PMID)."""
    results = await searcher.search(
        semantic_query.strip(),
        top_k=limit,
        entity_groups=entity_groups,
        expand_entities=expand_entities,
        scope=scope,
        include_content=False,
        stratify_by_pmid=True,
    )
    by_pmid: dict[str, PaperSearchResult] = {}
    for result in results:
        if result.pmid in by_pmid:
            continue
        by_pmid[result.pmid] = result
    return list(by_pmid.values())


async def _rank_pmids(
    vec_search: VectorSearchBase,
    semantic_query: str,
    pmids: list[str],
) -> pl.DataFrame:
    unfinished = pl.DataFrame({"PMID": pmids})

    results = []
    while unfinished.height > 0:
        print(f"Remaining PMIDs to rank: {unfinished.height:,}")
        batch = unfinished.head(500_000)
        rows = await vec_search.score_pmids(semantic_query, batch["PMID"].to_list(), limit=500_000)
        res = pl.DataFrame([{"PMID": pmid, "distance": dist} for pmid, dist in rows])
        unfinished = unfinished.join(res, on="PMID", how="anti")
        results.append(res)

    return pl.concat(results)


async def execute_retrieval(
    spec: RetrievalSpec,
    searcher: VectorSearchBase,
    *,
    scope: FilterScope,
) -> list[str]:
    """Execute a RetrievalSpec and return matching PMIDs (union of all filters).

    Multi-filter unions are ranked by:
    1) # of filters a PMID appears in (descending)
    2) max vector similarity across filters (descending)
    """
    if not spec.filters:
        return []

    assert isinstance(searcher, MilhouseVectorSearch), (
        "execute_retrieval currently requires MilhouseVectorSearch for filter execution"
    )

    pmid_to_hits = defaultdict(int)
    for f in tqdm(spec.filters, desc="Executing filters", unit="filter"):
        hits = await searcher.fetch_all_entity_hits(
            entity_groups=f.entity_groups,
        )
        for pmid, _ in hits:
            pmid_to_hits[pmid] += 1

    hit_df = pl.DataFrame([{"PMID": pmid, "num_hits": count} for pmid, count in pmid_to_hits.items()])

    fspec = spec.filters[0]
    vec_ranked_pmids = await _rank_pmids(
        vec_search=searcher,
        semantic_query=fspec.semantic_query,
        pmids=hit_df["PMID"].to_list(),
    )

    vec_ranked_pmids = vec_ranked_pmids.group_by("PMID").agg(
        pl.col.distance.mean().alias("mean_distance"), pl.col.distance.max().alias("max_distance")
    )

    pred_df = hit_df.join(vec_ranked_pmids, on="PMID", how="inner")

    pred_df = (
        pred_df.with_columns(
            [(pl.col(c).rank() / pl.len()).alias(f"{c}_pct") for c in ["num_hits", "mean_distance", "max_distance"]]
        )
        .with_columns(
            (
                0.5 * pl.col("num_hits_pct") + 0.35 * pl.col("mean_distance_pct") + 0.15 * pl.col("max_distance_pct")
            ).alias("score")
        )
        .sort("score", descending=True)
    )

    return pred_df.with_columns(pl.col("PMID").cast(pl.Utf8))["PMID"].to_list()
