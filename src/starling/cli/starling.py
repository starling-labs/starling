import json
from collections import Counter
from contextlib import AsyncExitStack

import dotenv
import hydra
import uvloop
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pydantic_ai import UsageLimits
from pydantic_ai.models import Model

from starling.cli.agent import (
    CorpusFilterDeps,
    RetrievalSpec,
    build_corpus_filter_agent,
    build_investigator_agent,
    build_validator_agent,
    execute_retrieval,
)
from starling.cli.conf.config import (
    StarlingAgentTuningConfig,
    StarlingConfig,
    StarlingExtractionConfig,
    StarlingMode,
    register_starling_config,
)
from starling.cli.extraction import build_extractor_agent, run_extraction
from starling.cli.extraction_types import ExtractionBatchResult, ExtractionSampleState
from starling.cli.utils import (
    ExtractionOutputWriter,
    ExtractionResumeState,
    load_extraction_resume_state,
    load_pmids_from_json,
    prune_obj,
    write_json_atomic,
)
from starling.infra.ch_client import build_clickhouse_engine
from starling.infra.ch_normalizer import CHEntityNormalizer
from starling.infra.logging import logfire_span, setup_logging
from starling.infra.milhouse_vector_search import MilhouseVectorSearch
from starling.infra.normalizer_base import EntityNormalizer
from starling.infra.vector_search_base import VectorSearchBase
from starling.models.llm import get_pydantic_model


async def _open_searcher(stack: AsyncExitStack) -> tuple[EntityNormalizer, MilhouseVectorSearch]:
    engine = await build_clickhouse_engine()
    normalizer: EntityNormalizer = CHEntityNormalizer(ch_client=engine)
    searcher = MilhouseVectorSearch(ch_client=engine, normalizer=normalizer)
    stack.push_async_callback(searcher.close)
    stack.push_async_callback(normalizer.close)
    return normalizer, searcher


async def _run_and_report_extraction(
    *,
    task: str,
    searcher: VectorSearchBase,
    model: Model,
    extract_config: StarlingExtractionConfig,
    output_path: str,
    retrieval_spec: RetrievalSpec | None = None,
    tuning_config: StarlingAgentTuningConfig | None = None,
    resume_state: ExtractionResumeState | None = None,
) -> ExtractionBatchResult:
    append = resume_state is not None

    if resume_state is not None:
        pmids = resume_state.remaining_pmids
        total_loaded = len(pmids)
        if extract_config.limit > 0:
            pmids = pmids[: extract_config.limit]
        if total_loaded != len(pmids):
            print(
                f"Loaded {total_loaded} PMIDs for extraction; "
                f"processing first {len(pmids)} due to extract.limit={extract_config.limit}."
            )
        else:
            print(f"Loaded {total_loaded} PMIDs for extraction.")
    elif extract_config.pmids_file:
        pmids_path = to_absolute_path(extract_config.pmids_file)
        pmids = load_pmids_from_json(pmids_path)
        total_loaded = len(pmids)
        if extract_config.limit > 0:
            pmids = pmids[: extract_config.limit]
        if not pmids:
            raise ValueError(f"No PMIDs found in PMID JSON: {pmids_path}")
        if total_loaded != len(pmids):
            print(
                f"Loaded {total_loaded} PMIDs from {pmids_path}; "
                f"extracting first {len(pmids)} due to extract.limit={extract_config.limit}."
            )
        else:
            print(f"Loaded {len(pmids)} PMIDs from {pmids_path}.")
    else:
        if retrieval_spec is None:
            raise ValueError("Retrieval spec is required when PMIDs are not provided directly.")
        if tuning_config is None:
            raise ValueError("Tuning config is required when PMIDs are not provided directly.")
        pmids = await execute_retrieval(
            spec=retrieval_spec,
            searcher=searcher,
            scope=tuning_config.filter_scope,
        )
    assert pmids is not None

    output_writer = ExtractionOutputWriter(output_path)
    if not append:
        output_writer.write_pmids_to_process(pmids)

    pmids = pmids[: extract_config.limit] if extract_config.limit > 0 else pmids
    extraction_result = await run_extraction(
        pmids=pmids,
        task=task,
        searcher=searcher,
        model=model,
        extract_config=extract_config,
        guidance=resume_state.guidance if resume_state is not None else None,
        on_result=output_writer.build_result_callback(append=append),
        output_writer=output_writer,
    )
    print("\n=== Extraction Summary ===")
    if resume_state is not None:
        print(
            f"Processed {extraction_result.total_papers} remaining papers this invocation "
            f"({len(resume_state.completed_pmids)} already present)."
        )
        completed = extraction_result.total_papers + len(resume_state.completed_pmids)
        print(f"Completed {completed}/{len(resume_state.pmids_to_process)} planned papers.")
    print(
        f"Extracted {extraction_result.total_extractions} records from "
        f"{extraction_result.successful_papers}/{extraction_result.total_papers} papers."
    )
    print(f"Status breakdown: {dict(Counter(result.status for result in extraction_result.results))}")
    print(f"\nWrote extraction results to {output_path}")
    return extraction_result


@logfire_span("run_starling")
async def run_starling(cfg: StarlingConfig) -> None:
    """Design a retrieval spec."""
    if not cfg.task:
        raise ValueError("mode=run requires `task=...`.")

    print(f"Run: {cfg.run_name}")
    print(f"Running Starling for task: {cfg.task}")
    print(f"Mode: {cfg.mode.value}")
    print(f"Main model: {cfg.models.main}")
    print(f"Validator model: {cfg.models.validator}")
    print(f"Investigator model: {cfg.models.investigator}")
    print(f"Extractor model: {cfg.models.extractor or cfg.models.main}")
    async with AsyncExitStack() as stack:
        normalizer, searcher = await _open_searcher(stack)

        pydantic_model = get_pydantic_model(cfg.models.main)

        validator_pydantic_model = get_pydantic_model(cfg.models.validator)
        validator = build_validator_agent(validator_pydantic_model)

        investigator_pydantic_model = get_pydantic_model(cfg.models.investigator)
        investigator = build_investigator_agent(investigator_pydantic_model)

        extractor_pydantic_model = get_pydantic_model(cfg.models.extractor or cfg.models.main)
        extractor = build_extractor_agent(extractor_pydantic_model)
        extraction_state = ExtractionSampleState()

        agent = build_corpus_filter_agent(
            pydantic_model,
            normalizer,
            include_extraction=True,
        )

        deps = CorpusFilterDeps(
            task=cfg.task,
            validator=validator,
            investigator=investigator,
            searcher=searcher,
            tuning=cfg.tuning,
            extractor=extractor,
            extraction_state=extraction_state,
        )
        result = await agent.run(
            cfg.task,
            deps=deps,
            usage_limits=UsageLimits(request_limit=200),
        )
        retrieval_spec = result.output

        print("\n=== Retrieval Specification ===")
        print(f"Filters: {len(retrieval_spec.filters)}")
        for i, f in enumerate(retrieval_spec.filters, 1):
            print(f"  Filter {i}:")
            print(f"    Entity groups: {f.entity_groups}")
            print(f"    Semantic query: {f.semantic_query}")
            if f.estimated_count is not None:
                print(f"    Estimated count: {f.estimated_count}")
            if f.precision_estimate is not None:
                print(f"    Precision: {f.precision_estimate:.1%}")

        spec_output = to_absolute_path(cfg.outputs.retrieval_spec)
        write_json_atomic(spec_output, retrieval_spec.model_dump())
        print(f"\nSaved RetrievalSpec to {spec_output}")

        # Print extraction sampling metrics if available
        if extraction_state.total_papers_sampled > 0:
            print("\n=== Extraction Sampling Metrics ===")
            print(f"Papers sampled: {extraction_state.total_papers_sampled}")
            print(f"Extraction rate: {extraction_state.overall_extraction_rate:.1%}")
            print(f"Total extractions: {extraction_state.total_extractions}")
            print(f"Unique entities: {extraction_state.total_unique_identifiers}")

        agent_run_output = to_absolute_path(cfg.outputs.agent_run)
        obj = json.loads(result.all_messages_json())
        obj = prune_obj(obj)
        write_json_atomic(agent_run_output, obj)  # ty:ignore[invalid-argument-type]


@logfire_span("run_extract")
async def run_extract(cfg: StarlingConfig) -> None:
    """Run extraction from a RetrievalSpec JSON file."""
    print(f"Run: {cfg.run_name}")
    resume_state = None
    task = cfg.task
    spec: RetrievalSpec | None = None

    if cfg.extract.resume_dir:
        if cfg.spec_file:
            raise ValueError("mode=extract with `extract.resume_dir` cannot also set `spec_file`.")
        if cfg.extract.pmids_file:
            raise ValueError("mode=extract with `extract.resume_dir` cannot also set `extract.pmids_file`.")

        resume_dir = to_absolute_path(cfg.extract.resume_dir)
        resume_state = load_extraction_resume_state(resume_dir)
        task = task or resume_state.guidance.task or resume_state.guidance.task_instantiation
        print(f"Resuming extraction from {resume_state.resume_dir}")
        print(f"  Loaded {len(resume_state.pmids_to_process)} planned PMIDs")
        print(f"  Already completed: {len(resume_state.completed_pmids)}")
        print(f"  Remaining: {len(resume_state.remaining_pmids)}")
        print(f"  Output: {resume_state.output_path}")
    else:
        if not cfg.spec_file:
            raise ValueError("mode=extract requires `spec_file=...` or `extract.resume_dir=...`.")
        if not task:
            raise ValueError("Fresh mode=extract runs require `task=...`.")

        spec_path = to_absolute_path(cfg.spec_file)
        with open(spec_path) as f:
            spec = RetrievalSpec.model_validate(json.load(f))

        print(f"Loaded RetrievalSpec from {spec_path}")
        print(f"  Filters: {len(spec.filters)}")
        for i, f in enumerate(spec.filters, 1):
            print(f"  Filter {i}: entities={len(f.entity_groups)} groups, query={f.semantic_query[:60]}...")
        if cfg.extract.pmids_file:
            print(f"  PMID override: {to_absolute_path(cfg.extract.pmids_file)}")

    assert task is not None
    print(f"Task: {task}")
    print(f"Model: {cfg.models.extractor or cfg.models.main}")
    print(f"Limit: {cfg.extract.limit} papers")
    print(f"Paper parallelism: {cfg.extract.parallelism}")
    print(f"Window parallelism: {cfg.extract.window_parallelism}")
    print()

    async with AsyncExitStack() as stack:
        _normalizer, searcher = await _open_searcher(stack)

        extractor_pydantic_model = get_pydantic_model(cfg.models.extractor or cfg.models.main)

        if resume_state is not None:
            if not resume_state.remaining_pmids:
                print("Nothing to resume; every PMID in pmids_to_process.json already has a result line.")
                return
            output_path = str(resume_state.output_path)
        else:
            output_path = to_absolute_path(cfg.outputs.extraction)
        await _run_and_report_extraction(
            retrieval_spec=spec,
            task=task,
            searcher=searcher,
            extract_config=cfg.extract,
            model=extractor_pydantic_model,
            output_path=output_path,
            tuning_config=cfg.tuning,
            resume_state=resume_state,
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: object) -> None:
    setup_logging()
    merged = OmegaConf.merge(OmegaConf.structured(StarlingConfig), cfg)
    cfg_obj = OmegaConf.to_object(merged)
    if not isinstance(cfg_obj, StarlingConfig):
        raise TypeError(f"Expected StarlingConfig from Hydra, got: {type(cfg_obj)}")

    try:
        match cfg_obj.mode:
            case StarlingMode.run:
                uvloop.run(run_starling(cfg_obj))
            case StarlingMode.extract:
                uvloop.run(run_extract(cfg_obj))
            case _:
                raise ValueError(f"Unknown mode: {cfg_obj.mode}")
    except KeyboardInterrupt:
        print("\nInterrupted.")


def entry() -> None:
    dotenv.load_dotenv()
    register_starling_config()
    main()


if __name__ == "__main__":
    entry()
