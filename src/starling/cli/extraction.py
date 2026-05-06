from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from importlib.resources import files
from typing import Any, Literal

import logfire
from jinja2 import Environment, FileSystemLoader
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models import Model

from starling.cli.conf.config import StarlingExtractionConfig
from starling.cli.extraction_types import (
    EXTRA_DETAILS_FIELD,
    ExtractionBatchResult,
    ExtractionGuidance,
    ExtractionRecord,
    ExtractorDeps,
    PaperExtractionResult,
    SchemaFieldSpec,
)
from starling.cli.schema import discover_schema_spec
from starling.cli.utils import (
    ExtractionOutputWriter,
    Gemma4Thinking,
    fetch_entity_hits,
    format_tagged_strings,
    iter_paragraph_windows,
)
from starling.infra.logging import logfire_span
from starling.infra.papers import Paper, PaperStore, TaggedString
from starling.infra.vector_search_base import VectorSearchBase
from starling.models.llm import get_pydantic_model
from starling.utils.limited import limited_as_completed
from starling.utils.paper_expansion import expand_around_idx

TEMPLATE_DIR = files("starling") / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)


OnResultCallback = Callable[[PaperExtractionResult], Awaitable[None] | None]


def _paper_result(
    pmid: str,
    *,
    status: Literal["success", "no_content", "failed"],
    extractions: list[ExtractionRecord] | None = None,
    error: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> PaperExtractionResult:
    return PaperExtractionResult(
        pmid=pmid,
        title=None,
        extractions=extractions or [],
        status=status,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def _load_paper(
    pmid: str,
    paper_store: PaperStore,
) -> Paper | PaperExtractionResult:
    try:
        paper = await paper_store.aget(pmid)
    except Exception as exc:
        return _paper_result(pmid, status="failed", error=f"{type(exc).__name__}: {exc}")
    if paper is None:
        return _paper_result(pmid, status="no_content", error="Paper not found in store.")
    return paper


def _coerce_field_value(value: Any) -> Any:
    """Leave scalars as-is; stringify composite structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict, tuple, set)):
        return str(value)
    return value


def _apply_schema_to_extracted(
    extracted: dict[str, Any],
    schema_spec: list[SchemaFieldSpec] | None,
) -> dict[str, Any]:
    if not schema_spec:
        return extracted

    schema_field_set = {spec.name for spec in schema_spec}
    include_extra_details = EXTRA_DETAILS_FIELD in schema_field_set
    structured_fields = [spec.name for spec in schema_spec if spec.name != EXTRA_DETAILS_FIELD]

    normalized = {field: _coerce_field_value(extracted.get(field)) for field in structured_fields}
    if not include_extra_details:
        return normalized

    parts: list[str] = []
    existing_extra = extracted.get(EXTRA_DETAILS_FIELD)
    if existing_extra is not None:
        parts.append(str(existing_extra))

    for key, value in extracted.items():
        if key in schema_field_set:
            continue
        parts.append(f"{key}: {value}")
    normalized[EXTRA_DETAILS_FIELD] = "; ".join(parts) if parts else None
    return normalized


def build_extractor_agent(model: Model) -> Agent[ExtractorDeps, list[ExtractionRecord]]:
    """Agent that extracts structured data from paper text."""

    def _get_extractor_instructions(ctx: RunContext[ExtractorDeps]) -> str:
        template = jinja_env.get_template("starling/extractor.jinja2")
        return template.render(
            task=ctx.deps.task,
            task_instantiation=ctx.deps.task_instantiation,
            pmid=ctx.deps.pmid,
            schema_spec=ctx.deps.schema_spec,
        )

    return Agent[ExtractorDeps, list[ExtractionRecord]](
        model=model,
        output_type=list[ExtractionRecord],
        deps_type=ExtractorDeps,
        retries=4,
        instructions=_get_extractor_instructions,
        capabilities=[Gemma4Thinking(), Thinking()],
    )


def _normalize_extractions(
    extractions: list[ExtractionRecord],
    pmid: str,
    schema_spec: list[SchemaFieldSpec] | None = None,
) -> list[ExtractionRecord]:
    return [
        record.model_copy(
            update={
                "pmid": pmid,
                "extraction_id": f"ext_{idx}",
                "extracted": _apply_schema_to_extracted(record.extracted, schema_spec),
            }
        )
        for idx, record in enumerate(extractions, start=1)
    ]


async def extract_from_paper(
    pmid: str,
    task: str,
    extractor: Agent[ExtractorDeps, list[ExtractionRecord]],
    paper_store: PaperStore,
    *,
    searcher: VectorSearchBase | None = None,
    tokens_per_paper: int,
    schema_spec: list[SchemaFieldSpec] | None = None,
    task_instantiation: str | None = None,
) -> PaperExtractionResult:
    """Extract from single paper using a token-bounded paper window (single call)."""
    paper = await _load_paper(pmid, paper_store)
    if isinstance(paper, PaperExtractionResult):
        return paper

    paragraphs = paper.flat_paragraphs()
    if not paragraphs:
        return _paper_result(pmid, status="no_content", error="Paper has no paragraph content.")

    center_idx = paragraphs[len(paragraphs) // 2].text.idx
    try:
        paper_window = expand_around_idx(
            paper,
            center_idx=center_idx,
            total_tokens=tokens_per_paper,
            drop_empty_sections=True,
        )
    except Exception as exc:
        return _paper_result(pmid, status="failed", error=f"Failed to build window: {type(exc).__name__}: {exc}")

    strings = paper_window.flat_tagged_strings(only_body=True)
    paragraph_idxs = {p.text.idx for p in paper_window.flat_paragraphs()}
    entity_hits = await fetch_entity_hits(searcher, pmid=pmid, paragraph_idxs=paragraph_idxs)
    paper_content = format_tagged_strings(strings, entity_hits=entity_hits)
    if not paper_content.strip():
        return _paper_result(pmid, status="no_content", error="No paragraph content in selected window.")

    deps = ExtractorDeps(task=task, task_instantiation=task_instantiation, pmid=pmid, schema_spec=schema_spec)
    try:
        result = await extractor.run(f"Paper PMID: {pmid}\n\n{paper_content}", deps=deps)
    except Exception as exc:
        return _paper_result(pmid, status="failed", error=f"{type(exc).__name__}: {exc}")

    usage = result.usage()
    extractions = _normalize_extractions(result.output or [], pmid, schema_spec=schema_spec)

    return _paper_result(
        pmid,
        status="success",
        extractions=extractions,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


async def _extract_window(
    *,
    pmid: str,
    window_idx: int,
    total_windows: int,
    start_para_idx: int,
    end_para_idx: int,
    window_strings: list[TaggedString],
    extractor: Agent[ExtractorDeps, list[ExtractionRecord]],
    deps: ExtractorDeps,
    window_semaphore: asyncio.Semaphore,
    entity_hits: dict[int, list[dict[str, object]]],
) -> tuple[list[ExtractionRecord], int, int]:
    """Returns (records, input_tokens, output_tokens)."""
    window_text = format_tagged_strings(window_strings, entity_hits=entity_hits)
    if not window_text.strip():
        return [], 0, 0

    prompt = (
        f"Paper PMID: {pmid}\n"
        f"Window {window_idx}/{total_windows} covering [p:{start_para_idx}] to [p:{end_para_idx}].\n"
        "Extract ONLY from the content in this window.\n\n"
        f"{window_text}"
    )
    try:
        async with window_semaphore:
            result = await extractor.run(prompt, deps=deps)
    except Exception:
        return [], 0, 0

    usage = result.usage()
    records = [record.model_copy(update={"pmid": pmid}) for record in result.output or []]
    return records, usage.input_tokens, usage.output_tokens


@logfire_span("starling_extract_from_paper_windows")
async def extract_from_paper_windows(
    pmid: str,
    task: str,
    extractor: Agent[ExtractorDeps, list[ExtractionRecord]],
    paper_store: PaperStore,
    window_paragraphs: int,
    *,
    window_semaphore: asyncio.Semaphore,
    searcher: VectorSearchBase | None = None,
    max_windows: int | None = None,
    schema_spec: list[SchemaFieldSpec] | None = None,
    task_instantiation: str | None = None,
) -> PaperExtractionResult:
    """Extract from a single paper by processing paragraph windows."""
    paper = await _load_paper(pmid, paper_store)
    if isinstance(paper, PaperExtractionResult):
        return paper

    paragraph_idx_set = {p.text.idx for p in paper.flat_paragraphs()}
    strings = paper.flat_tagged_strings(only_body=True)

    windows = iter_paragraph_windows(strings, paragraph_idx_set, window_paragraphs)
    if max_windows is not None and max_windows > 0:
        windows = windows[:max_windows]

    if not windows:
        return _paper_result(pmid, status="no_content", error="Paper has no windowable paragraph content.")

    entity_hits = await fetch_entity_hits(
        searcher,
        pmid=pmid,
        paragraph_idxs={
            ts.idx for _, _, window_strings in windows for ts in window_strings if ts.idx in paragraph_idx_set
        },
    )

    deps = ExtractorDeps(task=task, task_instantiation=task_instantiation, pmid=pmid, schema_spec=schema_spec)

    # Dedupe overlapping-window outputs using the extracted payload rather than the support text wording.
    by_key: dict[tuple[int, str | None, str], ExtractionRecord] = {}
    window_results = await asyncio.gather(
        *[
            _extract_window(
                pmid=pmid,
                window_idx=window_idx,
                total_windows=len(windows),
                start_para_idx=start_para_idx,
                end_para_idx=end_para_idx,
                window_strings=window_strings,
                extractor=extractor,
                deps=deps,
                window_semaphore=window_semaphore,
                entity_hits=entity_hits,
            )
            for window_idx, (start_para_idx, end_para_idx, window_strings) in enumerate(windows, start=1)
        ]
    )

    total_input_tokens = 0
    total_output_tokens = 0
    for window_records, win_input, win_output in window_results:
        total_input_tokens += win_input
        total_output_tokens += win_output
        for record in window_records:
            key = (
                record.support.paragraph_idx,
                record.global_identifier,
                json.dumps(record.extracted, ensure_ascii=False, sort_keys=True, default=str),
            )
            current = by_key.get(key)
            if current is None or record.confidence > current.confidence:
                by_key[key] = record

    merged = sorted(by_key.values(), key=lambda r: (r.support.paragraph_idx, -r.confidence))
    normalized = _normalize_extractions(merged, pmid, schema_spec=schema_spec)
    return _paper_result(
        pmid,
        status="success",
        extractions=normalized,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


async def run_extraction(
    pmids: list[str],
    task: str,
    searcher: VectorSearchBase,
    model: Model,
    *,
    extract_config: StarlingExtractionConfig,
    guidance: ExtractionGuidance | None = None,
    on_result: OnResultCallback | None = None,
    output_writer: ExtractionOutputWriter | None = None,
) -> ExtractionBatchResult:
    """Extract from multiple papers with backpressure.

    The run starts with a schema-discovery probe that runs until it collects
    `extract.schema_discovery_records` extraction records spanning at least
    `extract.schema_discovery_papers` unique PMIDs (hard-capped at the first 100 PMIDs),
    then applies the derived schema to every subsequent extraction for
    consistency.
    """
    if not pmids:
        task_instantiation = guidance.task_instantiation if guidance is not None else task
        schema_spec = guidance.schema_spec if guidance is not None else []
        if output_writer is not None:
            output_writer.write_guidance(
                task=guidance.task if guidance is not None else task,
                task_instantiation=task_instantiation,
                schema_spec=schema_spec,
            )
        print("No PMIDs to extract.")
        return ExtractionBatchResult.empty(
            task_instantiation=task_instantiation,
            schema_spec=schema_spec,
        )

    extractor = build_extractor_agent(model)

    if guidance is not None:
        schema_spec = guidance.schema_spec
        task_instantiation = guidance.task_instantiation or task
        print("Loaded existing extraction guidance; skipping schema discovery.")
    else:
        schema_deriver_model_name = extract_config.schema_deriver_model or model.model_name
        schema_deriver_model = get_pydantic_model(schema_deriver_model_name)
        schema_spec, task_instantiation = await discover_schema_spec(
            pmids=pmids,
            task=task,
            extractor=extractor,
            searcher=searcher,
            model=schema_deriver_model,
            extract_config=extract_config,
        )
    if schema_spec:
        print("Final schema spec:")
        for spec in schema_spec:
            print(f"  - {spec.name} ({spec.value_type}): {spec.description}")
            if spec.allowed_values:
                print(f"    Allowed values: {spec.allowed_values}")

    if output_writer is not None:
        output_writer.write_guidance(task=task, task_instantiation=task_instantiation, schema_spec=schema_spec)

    window_semaphore = asyncio.Semaphore(max(1, extract_config.window_parallelism))
    exit()

    async def yield_tasks():
        with logfire.span("starling_extraction"):
            for pmid in pmids:
                with logfire.span(f"extract_pmid_{pmid}"):
                    yield extract_from_paper_windows(
                        pmid=pmid,
                        task=task,
                        extractor=extractor,
                        paper_store=searcher.paper_store,
                        window_paragraphs=extract_config.window_paragraphs,
                        window_semaphore=window_semaphore,
                        searcher=searcher,
                        schema_spec=schema_spec,
                        task_instantiation=task_instantiation,
                    )

    results: list[PaperExtractionResult] = []
    total_papers = len(pmids)

    async for result in limited_as_completed(
        aws=yield_tasks(),
        in_flight=max(1, extract_config.parallelism),
        show_progress=True,
        return_exceptions=True,
        length_hint=len(pmids),
    ):
        if isinstance(result, BaseException):
            print(f"Error during extraction: {type(result).__name__}: {result}")
            continue

        results.append(result)

        if on_result is not None:
            try:
                maybe_awaitable = on_result(result)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except Exception as exc:
                print(f"Progress callback failed: {type(exc).__name__}: {exc}")

    return ExtractionBatchResult.from_results(
        task_instantiation=task_instantiation,
        results=results,
        total_papers=total_papers,
        schema_spec=schema_spec,
    )
