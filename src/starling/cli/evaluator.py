from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypedDict, get_args

import dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Thinking
from pydantic_ai.models import Model

from starling.cli.extraction_types import (
    ExtractionBatchResult,
    PaperExtractionResult,
    SchemaFieldSpec,
)
from starling.cli.utils import (
    Gemma4Thinking,
    _append_jsonl,
    _reset_jsonl_output,
    _validate_jsonl_output_path,
    format_paper_text,
    load_completed_pmids_from_jsonl,
)
from starling.infra.ch_client import build_clickhouse_engine
from starling.infra.logging import logfire_span, setup_logging
from starling.infra.milhouse_vector_search import MilhouseVectorSearch
from starling.infra.papers import PaperStore
from starling.infra.vector_search_base import VectorSearchBase
from starling.models.llm import get_pydantic_model
from starling.utils.limited import limited_as_completed

TEMPLATE_DIR = files("starling") / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)


Grade = Literal["pass", "flag", "fail"]
Dimensions = Literal[
    "support_fidelity",
    "task_relevance",
    "molecule_attribution",
    "label_correctness",
    "accuracy",
]

EVAL_DIMENSIONS: list[Dimensions] = list(get_args(Dimensions))
EVAL_BATCH_SIZE = 5


class DimensionDefinition(TypedDict):
    title: str
    question: str
    pass_definition: str
    flag_definition: str
    fail_definition: str


EVAL_DIMENSION_DEFINITIONS: dict[Dimensions, DimensionDefinition] = {
    "support_fidelity": {
        "title": "Support Fidelity",
        "question": "Is the `support_text` a faithful rendering of the source evidence from the paper?",
        "pass_definition": (
            "The support text is well grounded in the paper. The evidence may come from the cited paragraph, nearby "
            "paragraphs, tables, figures, or any other part of the provided paper text. Light rewriting, paraphrasing, "
            "and drawing reasonable conclusions from the stated facts are all fine, as long as the meaning and factual "
            "details remain faithful to the paper."
        ),
        "flag_definition": (
            "The support text is mostly grounded but omits a material qualifier that changes the strength or scope "
            "of the claim."
        ),
        "fail_definition": (
            "The support text cannot be supported by the paper text, materially distorts it, or fabricates "
            "details not present anywhere in the source."
        ),
    },
    "task_relevance": {
        "title": "Task Relevance",
        "question": "Does this extraction fall within the scope defined by the task?",
        "pass_definition": "The extraction captures exactly the type of data the task asks for.",
        "flag_definition": (
            "Borderline: the extraction is related but stretches the task scope "
            "(for example, a numeric proxy instead of an explicit task-specific label)."
        ),
        "fail_definition": (
            "The extraction is clearly out of scope "
            "(for example, a general physiology statement or a different property entirely)."
        ),
    },
    "molecule_attribution": {
        "title": "Molecule Attribution",
        "question": "Is the named molecule / entity actually the subject of the claim in this support text?",
        "pass_definition": (
            "The molecule / entity is the subject of the extracted claim and is either explicitly named in the support text, "
            "unambiguously established by immediate surrounding context, or referred to by a recognized synonym "
            "(e.g., noradrenaline for norepinephrine, acetylsalicylic acid for aspirin)."
        ),
        "flag_definition": (
            "The molecule / entity is not in the support text and the attribution from context is ambiguous."
        ),
        "fail_definition": (
            "The molecule / entity is not the subject of the claim, or it is absent from both the support text and its "
            "immediate context."
        ),
    },
    "label_correctness": {
        "title": "Label Correctness",
        "question": (
            "Is the primary label (the key outcome of the extraction) correct given the support text and its context?"
        ),
        "pass_definition": (
            "The primary outcome accurately reflects what the text states. "
            "Focus on the main result label, not on auxiliary or descriptive fields."
        ),
        "flag_definition": (
            "The primary label is defensible but a different allowed value would be equally or more appropriate."
        ),
        "fail_definition": "The primary label contradicts the text.",
    },
    "accuracy": {
        "title": "Accuracy",
        "question": "Is the core factual claim of the extraction faithful to the source text?",
        "pass_definition": (
            "The primary claim is supported by the text and nothing contradicts the source. "
            "This dimension is about factual faithfulness, not about whether the best possible schema field "
            "was chosen for a given value."
        ),
        "flag_definition": (
            "The primary claim requires a plausible but stretching inferential leap, or a secondary factual detail "
            "is debatable."
        ),
        "fail_definition": (
            "The extraction contradicts the source text or fabricates information not stated or inferable from the paper."
        ),
    },
}


class DimensionVerdict(BaseModel):
    grade: Grade
    reason: str | None = None


class ExtractionVerdict(BaseModel):
    """One grading verdict per extraction. Each dimension is an explicit field."""

    extraction_id: str

    support_fidelity: DimensionVerdict
    task_relevance: DimensionVerdict
    molecule_attribution: DimensionVerdict
    label_correctness: DimensionVerdict
    accuracy: DimensionVerdict

    re_extract: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_legacy_verdicts(cls, data: dict) -> dict:
        """Accept the old nested ``{"verdicts": {dim: ...}}`` format."""
        if isinstance(data, dict) and "verdicts" in data:
            verdicts = data.pop("verdicts")
            if isinstance(verdicts, dict):
                for dim in EVAL_DIMENSIONS:
                    if dim in verdicts and dim not in data:
                        data[dim] = verdicts[dim]
        return data


class PaperEvalResult(BaseModel):
    pmid: str
    extraction_verdicts: list[ExtractionVerdict]
    error: str | None = None


class EvalBatchResult(BaseModel):
    total_papers: int = 0
    total_extractions: int = 0
    results: list[PaperEvalResult] = Field(default_factory=list)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalBatchResult:
        jsonl_path = Path(path)
        results: list[PaperEvalResult] = []
        with jsonl_path.open() as f:
            for line_no, line in enumerate(f, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    results.append(PaperEvalResult.model_validate_json(payload))
                except ValidationError as exc:
                    raise ValueError(f"Invalid JSONL record in {jsonl_path} at line {line_no}") from exc

        return cls(
            total_papers=len(results),
            total_extractions=sum(len(result.extraction_verdicts) for result in results),
            results=results,
        )

    @classmethod
    def load(cls, path: str | Path) -> EvalBatchResult:
        """Load from JSON or JSONL based on file extension."""
        p = Path(path)
        if p.suffix == ".json":
            return cls.model_validate_json(p.read_bytes())
        return cls.from_jsonl(p)


@dataclass
class EvaluatorDeps:
    task_instantiation: str
    schema_spec: list[SchemaFieldSpec]
    template_path: str | None = None
    task_specific_guidance: str | None = None


type OnResultCallback = Callable[[PaperEvalResult], None]


def build_evaluator_agent(
    model: Model, retries: int = 3, max_tokens: int = 32768
) -> Agent[EvaluatorDeps, list[ExtractionVerdict]]:
    def _get_instructions(ctx: RunContext[EvaluatorDeps]) -> str:
        if ctx.deps.template_path:
            template = Environment(
                loader=FileSystemLoader(str(Path(ctx.deps.template_path).parent)),
                trim_blocks=True,
                lstrip_blocks=True,
            ).get_template(Path(ctx.deps.template_path).name)
        else:
            template = jinja_env.get_template("starling/evaluator.jinja2")
        return template.render(
            task_instantiation=ctx.deps.task_instantiation,
            schema_spec=ctx.deps.schema_spec,
            eval_dimensions=EVAL_DIMENSIONS,
            dimension_definitions=EVAL_DIMENSION_DEFINITIONS,
            task_specific_guidance=ctx.deps.task_specific_guidance,
        )

    return Agent[EvaluatorDeps, list[ExtractionVerdict]](
        model=model,
        output_type=list[ExtractionVerdict],
        deps_type=EvaluatorDeps,
        retries=retries,
        instructions=_get_instructions,
        # model_settings=ModelSettings(max_tokens=max_tokens),
        capabilities=[Gemma4Thinking(), Thinking()],
    )


def _format_extractions_for_prompt(paper_result: PaperExtractionResult) -> str:
    return json.dumps(
        [ext.model_dump() for ext in paper_result.extractions],
        indent=2,
        ensure_ascii=False,
        default=str,
    )


async def evaluate_paper(
    paper_result: PaperExtractionResult,
    evaluator: Agent[EvaluatorDeps, list[ExtractionVerdict]],
    deps: EvaluatorDeps,
    paper_store: PaperStore,
    *,
    searcher: VectorSearchBase,
    one_by_one: bool = False,
    context_radius: int = 5,
) -> PaperEvalResult:
    pmid = paper_result.pmid
    if not paper_result.extractions:
        return PaperEvalResult(pmid=pmid, extraction_verdicts=[])

    if one_by_one:
        all_verdicts: list[ExtractionVerdict] = []
        for ext in paper_result.extractions:
            single = paper_result.model_copy(update={"extractions": [ext]})
            ext_text = await format_paper_text(
                paper_store,
                pmid,
                target_idxs={ext.support.paragraph_idx},
                context_radius=context_radius,
                searcher=searcher,
            )
            if ext_text is None:
                continue
            prompt = (
                f"## Paper (PMID: {pmid})\n\n{ext_text}\n\n## Extractions\n\n{_format_extractions_for_prompt(single)}"
            )
            try:
                result = await evaluator.run(prompt, deps=deps)
                all_verdicts.extend(result.output or [])
            except Exception as exc:
                print(f"  Eval error for {pmid} {ext.extraction_id}: {type(exc).__name__}: {exc}")
        return PaperEvalResult(pmid=pmid, extraction_verdicts=all_verdicts)

    target_idxs = {ext.support.paragraph_idx for ext in paper_result.extractions}
    paper_text = await format_paper_text(
        paper_store,
        pmid,
        target_idxs=target_idxs,
        context_radius=context_radius,
        searcher=searcher,
    )
    if paper_text is None:
        return PaperEvalResult(pmid=pmid, extraction_verdicts=[], error="Paper not found in store.")

    all_verdicts: list[ExtractionVerdict] = []
    for i in range(0, len(paper_result.extractions), EVAL_BATCH_SIZE):
        chunk = paper_result.extractions[i : i + EVAL_BATCH_SIZE]
        chunk_paper = paper_result.model_copy(update={"extractions": chunk})
        prompt = f"## Paper (PMID: {pmid})\n\n{paper_text}\n\n## Extractions\n\n{_format_extractions_for_prompt(chunk_paper)}"
        try:
            result = await evaluator.run(prompt, deps=deps)
            all_verdicts.extend(result.output or [])
        except Exception as exc:
            print(f"  Eval error for {pmid} batch {i // EVAL_BATCH_SIZE}: {type(exc).__name__}: {exc}")

    return PaperEvalResult(pmid=pmid, extraction_verdicts=all_verdicts)


async def evaluate_papers(
    papers: list[PaperExtractionResult],
    evaluator: Agent[EvaluatorDeps, list[ExtractionVerdict]],
    deps: EvaluatorDeps,
    paper_store: PaperStore,
    *,
    parallelism: int = 10,
    context_radius: int = 5,
    one_by_one: bool = False,
    show_progress: bool = True,
    error_prefix: str = "Eval error",
    on_result: OnResultCallback | None = None,
    searcher: VectorSearchBase,
) -> tuple[list[PaperEvalResult], int]:
    results: list[PaperEvalResult] = []
    errors = 0
    tasks = [
        lambda paper=paper: evaluate_paper(
            paper,
            evaluator,
            deps,
            paper_store,
            one_by_one=one_by_one,
            context_radius=context_radius,
            searcher=searcher,
        )
        for paper in papers
    ]

    async for result in limited_as_completed(
        tasks,
        in_flight=parallelism,
        show_progress=show_progress,
        return_exceptions=True,
        length_hint=len(tasks),
    ):
        if isinstance(result, BaseException):
            errors += 1
            print(f"{error_prefix}: {type(result).__name__}: {result}", file=sys.stderr)
            continue
        results.append(result)
        if on_result is not None:
            try:
                on_result(result)
            except Exception as exc:
                print(f"Eval output callback failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    return results, errors


@logfire_span("run_evaluation")
async def run_evaluation(
    batch: ExtractionBatchResult,
    model: Model,
    paper_store: PaperStore,
    *,
    sample_size: int | None = None,
    parallelism: int = 10,
    context_radius: int = 5,
    show_progress: bool = True,
    error_prefix: str = "Eval error",
    on_result: OnResultCallback | None = None,
    one_by_one: bool = False,
    searcher: VectorSearchBase,
) -> tuple[EvalBatchResult, int]:
    papers = [result for result in batch.results if result.status == "success" and result.extractions]
    if sample_size is not None and sample_size < len(papers):
        papers = random.sample(papers, sample_size)

    eval_results, errors = await evaluate_papers(
        papers,
        build_evaluator_agent(model),
        EvaluatorDeps(
            task_instantiation=batch.task_instantiation,
            schema_spec=batch.schema_spec or [],
        ),
        paper_store,
        parallelism=parallelism,
        context_radius=context_radius,
        one_by_one=one_by_one,
        show_progress=show_progress,
        error_prefix=error_prefix,
        on_result=on_result,
        searcher=searcher,
    )
    return (
        EvalBatchResult(
            total_papers=len(eval_results),
            total_extractions=sum(len(result.extraction_verdicts) for result in eval_results),
            results=eval_results,
        ),
        errors,
    )


def tally_grades(
    eval_results: list[PaperEvalResult],
) -> tuple[dict[str, dict[Grade, int]], int, list[tuple[str, str, str, str]]]:
    """Return grade counts, evaluated extraction count, and failing verdicts by dimension."""
    counts: dict[str, dict[Grade, int]] = {d: {"pass": 0, "flag": 0, "fail": 0} for d in EVAL_DIMENSIONS}
    total = 0
    failures: list[tuple[str, str, str, str]] = []

    for paper in eval_results:
        for verdict in paper.extraction_verdicts:
            total += 1
            for dim in EVAL_DIMENSIONS:
                dv: DimensionVerdict = getattr(verdict, dim)
                counts[dim][dv.grade] += 1
                if dv.grade == "fail" and dv.reason:
                    failures.append((paper.pmid, verdict.extraction_id, dim, dv.reason))

    return counts, total, failures


def _molecule_stats(
    eval_results: list[PaperEvalResult],
    extraction_batch: ExtractionBatchResult,
) -> tuple[int, int]:
    """Return (total unique molecules, unique molecules with no fails)."""
    # Build lookup: (pmid, extraction_id) → global_identifier
    id_lookup: dict[tuple[str, str], str] = {}
    for paper in extraction_batch.results:
        for ext in paper.extractions:
            if ext.extraction_id and ext.global_identifier:
                id_lookup[(paper.pmid, ext.extraction_id)] = ext.global_identifier

    all_molecules: set[str] = set()
    failed_molecules: set[str] = set()
    for paper in eval_results:
        for verdict in paper.extraction_verdicts:
            gid = id_lookup.get((paper.pmid, verdict.extraction_id))
            if not gid:
                continue
            all_molecules.add(gid)
            if any(getattr(verdict, dim).grade == "fail" for dim in EVAL_DIMENSIONS):
                failed_molecules.add(gid)

    return len(all_molecules), len(all_molecules - failed_molecules)


def print_summary(
    eval_batch: EvalBatchResult,
    extraction_batch: ExtractionBatchResult | None = None,
) -> None:
    counts, total, failures = tally_grades(eval_batch.results)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation Summary ({total} extractions from {eval_batch.total_papers} papers)")
    print(f"{'=' * 60}\n")

    for dim in EVAL_DIMENSIONS:
        c = counts[dim]
        evaluated = c["pass"] + c["flag"] + c["fail"]
        if evaluated == 0:
            continue
        pass_pct = 100 * c["pass"] / evaluated
        flag_pct = 100 * c["flag"] / evaluated
        fail_pct = 100 * c["fail"] / evaluated
        print(
            f"  {dim:25s}  pass={c['pass']:5d} ({pass_pct:5.1f}%)  flag={c['flag']:5d} ({flag_pct:5.1f}%)  fail={c['fail']:5d} ({fail_pct:5.1f}%)"
        )

    # Extraction-level aggregate grades
    full_pass = 0
    soft_pass = 0
    hard_fail = 0
    recoverable = 0
    for paper in eval_batch.results:
        for verdict in paper.extraction_verdicts:
            grades = [getattr(verdict, dim).grade for dim in EVAL_DIMENSIONS]
            if all(g == "pass" for g in grades):
                full_pass += 1
            elif any(g == "fail" for g in grades):
                hard_fail += 1
                if verdict.re_extract:
                    recoverable += 1
            else:
                soft_pass += 1

    if total:
        print()
        print(f"  {'Full pass (all pass)':25s}  {full_pass:5d} ({100 * full_pass / total:5.1f}%)")
        print(f"  {'Soft pass (no fail)':25s}  {soft_pass:5d} ({100 * soft_pass / total:5.1f}%)")
        print(f"  {'Hard fail (any fail)':25s}  {hard_fail:5d} ({100 * hard_fail / total:5.1f}%)")
        print(f"  {'Recoverable (re_extract)':25s}  {recoverable:5d} ({100 * recoverable / total:5.1f}%)")

    print()

    if extraction_batch is not None:
        total_molecules, no_fail_molecules = _molecule_stats(eval_batch.results, extraction_batch)
        print(f"  Unique molecules: {total_molecules}  (no fails: {no_fail_molecules})\n")

    if failures:
        print("  Sample failures (showing up to 10):\n")
        for pmid, ext_id, dim, reason in failures[:10]:
            print(f"    PMID={pmid} {ext_id} [{dim}]: {reason}")
        print()


def _build_eval_checkpoint_callback(output_path: str, *, append: bool = False) -> OnResultCallback:
    _validate_jsonl_output_path(output_path)
    if append:
        print(f"Appending evaluation results incrementally to {output_path}")
    else:
        _reset_jsonl_output(output_path)
        print(f"Writing evaluation results incrementally to {output_path}")

    def _checkpoint(result: PaperEvalResult) -> None:
        _append_jsonl(output_path, result.model_dump(mode="json"))

    return _checkpoint


async def _main() -> None:
    dotenv.load_dotenv()
    setup_logging(
        logfire_min_level="warn",
        console=False,
    )

    parser = argparse.ArgumentParser(description="Evaluate extraction quality")
    parser.add_argument("--extractions", required=True, help="Path to extractions.jsonl (or legacy extractions.json)")
    parser.add_argument("--model", default="openrouter:openai/gpt-oss-120b", help="Evaluator model")
    parser.add_argument("--sample", type=int, default=None, help="Number of papers to sample (default: all)")
    parser.add_argument("--parallelism", type=int, default=10)
    parser.add_argument("--context-radius", type=int, default=5, help="Paragraphs of context around each extraction")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <input_dir>/eval_results.jsonl)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--one-by-one",
        action="store_true",
        help="Evaluate each extraction individually instead of batching all extractions per paper",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    batch = ExtractionBatchResult.load(args.extractions)
    print(f"Loaded {len(batch.results)} papers, {batch.total_extractions} extractions")
    print(f"Task: {(batch.task_instantiation)[:200]}...")

    output_path = args.output or str(Path(args.extractions).parent / "eval_results.jsonl")

    # Resume support: load already-evaluated PMIDs and skip them
    completed_pmids = load_completed_pmids_from_jsonl(output_path)
    append = bool(completed_pmids)
    if completed_pmids:
        print(f"Resuming: {len(completed_pmids)} papers already evaluated in {output_path}")
        pre_filter = len(batch.results)
        batch.results = [r for r in batch.results if r.pmid not in completed_pmids]
        print(f"Skipping {pre_filter - len(batch.results)} already-evaluated papers, {len(batch.results)} remaining")

    model = get_pydantic_model(args.model)
    paper_store = PaperStore()

    ch_client = await build_clickhouse_engine()
    searcher = MilhouseVectorSearch(ch_client)

    eval_batch, _ = await run_evaluation(
        batch,
        model,
        paper_store,
        sample_size=args.sample,
        parallelism=args.parallelism,
        context_radius=args.context_radius,
        on_result=_build_eval_checkpoint_callback(output_path, append=append),
        one_by_one=args.one_by_one,
        searcher=searcher,
    )

    # Reload full results for summary
    if append:
        eval_batch = EvalBatchResult.from_jsonl(output_path)

    print_summary(eval_batch)
    print(f"Wrote {eval_batch.total_papers} paper eval results to {output_path}")


def main() -> None:
    asyncio.run(_main())


def summary() -> None:
    parser = argparse.ArgumentParser(description="Print evaluation summary from results (JSON or JSONL)")
    parser.add_argument("results", help="Path to eval_results.jsonl or eval_results.json")
    parser.add_argument("--extractions", default=None, help="Path to extractions file (enables molecule stats)")
    args = parser.parse_args()
    eval_batch = EvalBatchResult.load(args.results)
    extraction_batch = ExtractionBatchResult.load(args.extractions) if args.extractions else None
    print_summary(eval_batch, extraction_batch)


if __name__ == "__main__":
    main()
