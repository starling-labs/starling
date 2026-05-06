from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import logfire
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, RunContext, Tool
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models import Model

from starling.cli.conf.config import StarlingExtractionConfig
from starling.cli.evaluator import (
    EVAL_DIMENSIONS,
    EvaluatorDeps,
    ExtractionVerdict,
    PaperEvalResult,
    build_evaluator_agent,
    evaluate_paper,
    tally_grades,
)
from starling.cli.extraction_types import (
    EXTRA_DETAILS_FIELD,
    ExtractionGuidance,
    ExtractionRecord,
    ExtractorDeps,
    PaperExtractionResult,
    SchemaFieldSpec,
)
from starling.infra.logging import logfire_span
from starling.infra.vector_search_base import VectorSearchBase
from starling.utils.limited import limited_as_completed

TEMPLATE_DIR = files("starling") / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)

_NON_IDENTIFIER_CHARS = re.compile(r"[^a-z0-9_]+")


def _normalize_schema_field_name(name: str) -> str | None:
    raw = name.strip().lower()
    if not raw:
        return None
    raw = raw.replace("-", "_").replace(" ", "_")
    raw = _NON_IDENTIFIER_CHARS.sub("_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        return None
    if raw[0].isdigit():
        raw = f"field_{raw}"
    return raw


def _normalize_schema_value_type(value_type: str | None) -> str:
    normalized = (value_type or "").strip().lower()
    if normalized in {"number", "numeric", "float", "double", "int", "integer", "decimal"}:
        return "number"
    if normalized in {"boolean", "bool"}:
        return "boolean"
    return "string"


def finalize_schema_spec(specs: list[SchemaFieldSpec], max_fields: int) -> list[SchemaFieldSpec]:
    """Normalize, deduplicate, and enforce required `extra_details` + max field count."""
    if max_fields <= 0:
        return []

    structured_limit = max(0, max_fields - 1)
    normalized_specs: list[SchemaFieldSpec] = []
    seen: set[str] = set()

    for spec in specs:
        name = _normalize_schema_field_name(spec.name)
        if not name or name == EXTRA_DETAILS_FIELD or name in seen:
            continue
        seen.add(name)

        description = (spec.description or "").strip() or f"Extract `{name}` from the passage, if explicitly stated."
        allowed_values = [str(v).strip() for v in spec.allowed_values if str(v).strip()]

        normalized_specs.append(
            SchemaFieldSpec(
                name=name,
                description=description,
                value_type=_normalize_schema_value_type(spec.value_type),
                allowed_values=allowed_values,
            )
        )
        if len(normalized_specs) >= structured_limit:
            break

    normalized_specs.append(
        SchemaFieldSpec(
            name=EXTRA_DETAILS_FIELD,
            description="Free-form string for additional details that do not fit other top-level fields.",
            value_type="string",
            allowed_values=[],
        )
    )
    return normalized_specs


_QUALITY_DIMS = {"task_relevance", "label_correctness", "accuracy"}


def pick_best_validated_schema(
    result: AgentRunResult[ExtractionGuidance],
) -> tuple[list[SchemaFieldSpec], str] | None:
    """Select the validate_schema call whose results have the lowest combined fail rate.

    Sums fail_pct across task_relevance, label_correctness, and accuracy for each
    validate_schema tool call and returns the (schema_spec, task_instantiation) pair
    with the smallest sum.  Returns None when no successful validation calls exist.
    """
    tool_returns: dict[str, str | dict[str, Any]] = {}
    for msg in result.all_messages():
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_name == "validate_schema"
                    and part.outcome == "success"
                ):
                    tool_returns[part.tool_call_id] = part.content  # ty:ignore[invalid-assignment]

    candidates: list[tuple[float, list[SchemaFieldSpec], str]] = []
    for msg in result.all_messages():
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolCallPart) or part.tool_name != "validate_schema":
                continue
            ret = tool_returns.get(part.tool_call_id)
            if ret is None:
                continue
            args = part.args_as_dict()
            vr = ret if isinstance(ret, ValidationResult) else ValidationResult.model_validate(ret)
            fail_sum = sum(d.fail_pct for d in vr.dimensions if d.dimension in _QUALITY_DIMS)
            spec = [SchemaFieldSpec.model_validate(s) for s in args["schema_spec"]]
            candidates.append((fail_sum, spec, args["task_instantiation"]))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    best_fail, best_spec, best_task = candidates[0]
    print(f"Selected schema with lowest fail sum ({best_fail:.1f}%) from {len(candidates)} candidate(s).")
    return best_spec, best_task


@dataclass
class SchemaDeriveDeps:
    task: str
    max_fields: int

    # These resources are only exercised when schema validation is enabled.
    extractor: Agent[ExtractorDeps, list[ExtractionRecord]]
    evaluator_model: Model
    searcher: VectorSearchBase
    validation_enabled: bool = True
    validation_pmids: list[str] = field(default_factory=list)
    paper_parallelism: int = 10
    window_paragraphs: int = 5
    window_semaphore: asyncio.Semaphore | None = None
    max_validation_rounds: int = 1


class DimensionStats(BaseModel):
    dimension: str
    pass_pct: float
    flag_pct: float
    fail_pct: float


class IssueExample(BaseModel):
    reason: str
    extraction: dict[str, Any] = Field(default_factory=dict)


class DimensionIssues(BaseModel):
    dimension: str
    total: int
    examples: list[IssueExample]


class ValidationResult(BaseModel):
    papers: int
    extractions: int
    evaluated: int
    all_pass: int
    dimensions: list[DimensionStats]
    failures: list[DimensionIssues]
    flags: list[DimensionIssues]


MAX_ISSUE_EXAMPLES = 4


def _issue_example(
    ext_lookup: dict[tuple[str, str], ExtractionRecord],
    pmid: str,
    ext_id: str,
    reason: str,
) -> IssueExample:
    extraction = ext_lookup.get((pmid, ext_id))
    return IssueExample(reason=reason, extraction=extraction.model_dump() if extraction else {})


def _group_issues(
    rows: list[tuple[str, str, str, str]],
    ext_lookup: dict[tuple[str, str], ExtractionRecord],
) -> list[DimensionIssues]:
    by_dimension: dict[str, list[tuple[str, str, str]]] = {}
    for pmid, ext_id, dim, reason in rows:
        by_dimension.setdefault(dim, []).append((pmid, ext_id, reason))
    return [
        DimensionIssues(
            dimension=dim,
            total=len(items),
            examples=[_issue_example(ext_lookup, *item) for item in items[:MAX_ISSUE_EXAMPLES]],
        )
        for dim, items in by_dimension.items()
    ]


def _build_validation_result(
    extraction_results: list[PaperExtractionResult],
    eval_results: list[PaperEvalResult],
) -> ValidationResult:
    ext_lookup = {
        (paper_result.pmid, extraction.extraction_id or ""): extraction
        for paper_result in extraction_results
        for extraction in paper_result.extractions
    }
    grade_counts, total_evaluated, failures = tally_grades(eval_results)
    flags = [
        (paper_eval.pmid, verdict.extraction_id, dim, getattr(verdict, dim).reason)
        for paper_eval in eval_results
        for verdict in paper_eval.extraction_verdicts
        for dim in EVAL_DIMENSIONS
        if getattr(verdict, dim).grade == "flag" and getattr(verdict, dim).reason
    ]

    def _dimension_stats(dimension: str, counts: dict[str, int]) -> DimensionStats:
        evaluated = counts["pass"] + counts["flag"] + counts["fail"]
        if evaluated == 0:
            return DimensionStats(
                dimension=dimension,
                pass_pct=0.0,
                flag_pct=0.0,
                fail_pct=0.0,
            )

        return DimensionStats(
            dimension=dimension,
            pass_pct=round(100 * counts["pass"] / evaluated, 1),
            flag_pct=round(100 * counts["flag"] / evaluated, 1),
            fail_pct=round(100 * counts["fail"] / evaluated, 1),
        )

    return ValidationResult(
        papers=len(extraction_results),
        extractions=sum(len(result.extractions) for result in extraction_results),
        evaluated=total_evaluated,
        all_pass=sum(
            1
            for paper_eval in eval_results
            for verdict in paper_eval.extraction_verdicts
            if all(getattr(verdict, dim).grade == "pass" for dim in EVAL_DIMENSIONS)
        ),
        dimensions=[
            _dimension_stats(dim, c)  # ty:ignore[invalid-argument-type]
            for dim, c in grade_counts.items()
            if any(c.values())
        ],
        failures=_group_issues(failures, ext_lookup),
        flags=_group_issues(flags, ext_lookup),  # ty:ignore[invalid-argument-type]
    )


@logfire_span("starling_paper_validation")
async def _extract_and_evaluate_validation_paper(
    pmid: str,
    *,
    task: str,
    task_instantiation: str,
    schema_spec: list[SchemaFieldSpec],
    extractor: Agent[ExtractorDeps, list[ExtractionRecord]],
    evaluator: Agent[EvaluatorDeps, list[ExtractionVerdict]],
    evaluator_deps: EvaluatorDeps,
    searcher: VectorSearchBase,
    window_paragraphs: int,
    window_semaphore: asyncio.Semaphore,
) -> tuple[PaperExtractionResult, PaperEvalResult] | None:
    from starling.cli.extraction import extract_from_paper_windows

    extraction_result = await extract_from_paper_windows(
        pmid=pmid,
        task=task,
        extractor=extractor,
        paper_store=searcher.paper_store,
        window_paragraphs=window_paragraphs,
        window_semaphore=window_semaphore,
        searcher=searcher,
        schema_spec=schema_spec,
        task_instantiation=task_instantiation,
    )
    if extraction_result.status != "success" or not extraction_result.extractions:
        return None

    return extraction_result, await evaluate_paper(
        extraction_result,
        evaluator,
        evaluator_deps,
        searcher.paper_store,
        searcher=searcher,
    )


async def _validate_schema_tool(
    ctx: RunContext[SchemaDeriveDeps],
    task_instantiation: str,
    schema_spec: list[SchemaFieldSpec],
) -> ValidationResult:
    """Run test extractions with the proposed schema on validation papers and evaluate quality.

    Call this to check whether your task_instantiation and schema produce high-quality
    extractions before finalizing. Review the evaluation summary and revise if needed.
    """
    deps = ctx.deps
    finalized = finalize_schema_spec(schema_spec, max_fields=deps.max_fields)
    evaluator = build_evaluator_agent(deps.evaluator_model)
    evaluator_deps = EvaluatorDeps(task_instantiation=task_instantiation, schema_spec=finalized)
    window_semaphore = deps.window_semaphore or asyncio.Semaphore(1)

    extraction_results: list[PaperExtractionResult] = []
    eval_results: list[PaperEvalResult] = []
    validation_tasks = (
        lambda pmid=pmid: _extract_and_evaluate_validation_paper(
            pmid=pmid,
            task=deps.task,
            task_instantiation=task_instantiation,
            schema_spec=finalized,
            extractor=deps.extractor,
            evaluator=evaluator,
            evaluator_deps=evaluator_deps,
            searcher=deps.searcher,
            window_paragraphs=deps.window_paragraphs,
            window_semaphore=window_semaphore,
        )
        for pmid in deps.validation_pmids
    )
    in_flight = max(1, min(deps.paper_parallelism, len(deps.validation_pmids)))
    async for result in limited_as_completed(
        validation_tasks,
        in_flight=in_flight,
        show_progress=False,
        return_exceptions=True,
        length_hint=len(deps.validation_pmids),
    ):
        if isinstance(result, BaseException) or result is None:
            continue
        extraction_result, eval_result = result
        extraction_results.append(extraction_result)
        eval_results.append(eval_result)

    if not extraction_results:
        return ValidationResult(papers=0, extractions=0, evaluated=0, all_pass=0, dimensions=[], failures=[], flags=[])

    return _build_validation_result(extraction_results, eval_results)


def build_schema_deriver_agent(
    model: Model, *, validation_enabled: bool
) -> Agent[SchemaDeriveDeps, ExtractionGuidance]:
    """Agent that derives extractor guidance (task instantiation + compact schema) from raw extraction examples."""

    def _get_schema_deriver_instructions(ctx: RunContext[SchemaDeriveDeps]) -> str:
        template = jinja_env.get_template("starling/schema_deriver.jinja2")
        return template.render(
            task=ctx.deps.task,
            max_fields=ctx.deps.max_fields,
            max_validation_rounds=ctx.deps.max_validation_rounds,
            schema_validation_enabled=ctx.deps.validation_enabled,
        )

    tools: list[Tool[SchemaDeriveDeps]] = []
    if validation_enabled:
        tools.append(Tool(_validate_schema_tool, name="validate_schema"))

    agent = Agent[SchemaDeriveDeps, ExtractionGuidance](
        model=model,
        output_type=ExtractionGuidance,
        deps_type=SchemaDeriveDeps,
        retries=3,
        instructions=_get_schema_deriver_instructions,
        tools=tools,
    )

    return agent


@logfire_span("starling_schema_discovery")
async def discover_schema_spec(
    pmids: list[str],
    task: str,
    extractor: Agent[ExtractorDeps, list[ExtractionRecord]],
    searcher: VectorSearchBase,
    *,
    model: Model,
    extract_config: StarlingExtractionConfig,
) -> tuple[list[SchemaFieldSpec], str]:
    """Probe early papers to derive a consistent extraction schema for the run."""
    from starling.cli.extraction import extract_from_paper_windows

    sample_papers = max(1, extract_config.schema_discovery_papers)
    validation_enabled = extract_config.schema_validate and extract_config.schema_validation_papers > 0
    validation_papers = max(1, extract_config.schema_validation_papers) if validation_enabled else 0
    target_papers_with_extractions = sample_papers + validation_papers

    hard_cap_papers = min(100, len(pmids))
    probed_pmids = pmids[:hard_cap_papers]

    pmids_with_extractions: set[str] = set()
    collected: list[ExtractionRecord] = []
    papers_attempted = 0

    with logfire.span("schema_discovery_probe"):
        window_semaphore = asyncio.Semaphore(max(1, extract_config.window_parallelism))
        probe_tasks = [
            lambda pmid=pmid: extract_from_paper_windows(
                pmid=pmid,
                task=task,
                extractor=extractor,
                paper_store=searcher.paper_store,
                window_paragraphs=extract_config.window_paragraphs,
                window_semaphore=window_semaphore,
                searcher=searcher,
                schema_spec=None,
            )
            for pmid in probed_pmids
        ]
        parallelism = min(extract_config.parallelism, 100)
        in_flight = min(max(1, parallelism), len(probe_tasks))
        async for result in limited_as_completed(
            probe_tasks,
            in_flight=in_flight,
            return_exceptions=True,
            show_progress=False,
            length_hint=len(probe_tasks),
        ):
            if isinstance(result, BaseException):
                continue
            papers_attempted += 1
            if result.status == "success" and result.extractions:
                pmids_with_extractions.add(result.pmid)
                collected.extend(result.extractions)

            if (
                len(collected) >= extract_config.schema_discovery_records
                and len(pmids_with_extractions) >= target_papers_with_extractions
            ):
                break

    by_pmid: dict[str, list[ExtractionRecord]] = {}
    for record in collected:
        by_pmid.setdefault(record.pmid, []).append(record)
    for records in by_pmid.values():
        records.sort(key=lambda r: r.confidence, reverse=True)

    pmid_rank = {pmid: idx for idx, pmid in enumerate(probed_pmids)}
    ordered_pmids = sorted(by_pmid.keys(), key=lambda pmid: pmid_rank.get(pmid, len(probed_pmids)))

    sample_pmids = ordered_pmids[:sample_papers]
    if sum(len(by_pmid[pmid]) for pmid in sample_pmids) < extract_config.schema_discovery_records:
        sample_pmids = ordered_pmids

    top_records: list[ExtractionRecord] = []
    offset = 0
    while len(top_records) < extract_config.schema_discovery_records:
        progressed = False
        for pmid in sample_pmids:
            records = by_pmid[pmid]
            if offset < len(records):
                top_records.append(records[offset])
                progressed = True
                if len(top_records) >= extract_config.schema_discovery_records:
                    break
        if not progressed:
            break
        offset += 1

    sample_records: list[dict[str, Any]] = [
        {
            "pmid": record.pmid,
            "paragraph_idx": record.support.paragraph_idx,
            "confidence": record.confidence,
            "support_text": record.support.support_text[:400],
            "extracted": record.extracted,
        }
        for record in top_records[: extract_config.schema_discovery_records]
    ]

    sample_pmid_set = {r["pmid"] for r in sample_records}
    validation_pmids: list[str] = []
    if validation_enabled:
        holdout_validation_pmids = [pmid for pmid in ordered_pmids if pmid not in sample_pmid_set][:validation_papers]
        discovery_validation_pmids = [pmid for pmid in ordered_pmids if pmid in sample_pmid_set][
            : validation_papers // 2
        ]
        validation_pmids = holdout_validation_pmids + discovery_validation_pmids
        if not validation_pmids:
            validation_pmids = sample_pmids[:validation_papers] or ordered_pmids[:1]

    schema_deriver = build_schema_deriver_agent(model, validation_enabled=validation_enabled)
    deps = SchemaDeriveDeps(
        task=task,
        max_fields=extract_config.schema_max_fields,
        validation_enabled=validation_enabled,
        extractor=extractor,
        evaluator_model=extractor.model,  # ty: ignore[invalid-argument-type]
        searcher=searcher,
        validation_pmids=validation_pmids,
        paper_parallelism=extract_config.parallelism,
        window_paragraphs=extract_config.window_paragraphs,
        window_semaphore=asyncio.Semaphore(max(1, extract_config.window_parallelism)),
    )
    if validation_enabled:
        print(f"Schema validation will use {len(validation_pmids)} papers.")
    else:
        print("Schema validation disabled; deriving schema from discovery samples only.")

    prompt = json.dumps({"sample_records": sample_records}, ensure_ascii=False, indent=2, default=str)
    prompt = f"```json\n{prompt}\n```"
    result = await schema_deriver.run(prompt, deps=deps)

    # Crawl the tool calls to try and find the actual best one because I don't
    # trust the agent to actually give me the best schema it found at the end of the run.
    best = pick_best_validated_schema(result)
    if best is not None:
        proposed, task_instantiation = best
    else:
        guidance = result.output
        proposed = guidance.schema_spec
        task_instantiation = guidance.task_instantiation

    schema_spec = finalize_schema_spec(proposed, max_fields=extract_config.schema_max_fields)
    schema_fields = [spec.name for spec in schema_spec]
    print(
        f"Derived schema from {len(top_records)} records across {len(sample_pmids)} papers "
        f"(probed {papers_attempted}/{len(probed_pmids)}): {schema_fields}"
    )

    return schema_spec, task_instantiation
