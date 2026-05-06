from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import coolname
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from starling.infra.entity_filter import FilterScope


class StarlingMode(StrEnum):
    run = "run"
    extract = "extract"


@dataclass
class StarlingModelsConfig:
    main: str = "openai-responses:gpt-5.2"

    sub: str = "openrouter:openai/gpt-oss-120b:exacto"
    validator: str = "openrouter:openai/gpt-oss-120b:exacto"
    investigator: str = "openrouter:openai/gpt-oss-120b:exacto"
    extractor: str | None = "openrouter:openai/gpt-oss-120b:exacto"


@dataclass
class StarlingOutputsConfig:
    output_root: str = "runs"
    retrieval_spec: str = "retrieval_spec.json"
    agent_run: str = "agent_run.json"
    extraction: str = "extractions.jsonl"


@dataclass
class StarlingExtractionConfig:
    pmids_file: str | None = None
    resume_dir: str | None = None

    limit: int = 100
    parallelism: int = 10
    window_parallelism: int = 10
    window_paragraphs: int = 5
    schema_discovery_records: int = 10
    schema_discovery_papers: int = 15
    schema_validate: bool = True
    schema_validation_papers: int = 20
    schema_max_fields: int = 9

    schema_deriver_model: str = "openai-responses:gpt-5.2"  # Use the honker


@dataclass
class StarlingAgentTuningConfig:
    filter_scope: FilterScope = FilterScope.paragraph

    target_precision: float = 0.80
    target_recall_gap: float = 0.15
    target_extraction_rate: float = 0.50
    marginal_recall_gain_threshold: float = 0.05
    max_sub_filters: int = 8
    marginal_gain_base_k: int = 400
    marginal_gain_candidate_k: int = 200
    marginal_gain_validation_sample_size: int = 50

    max_examples_returned: int = 10
    max_snippet_entities: int = 25

    # Extraction sampling tool defaults (agent may override at call-time).
    extraction_sample_default_size: int = 100
    extraction_sample_parallelism: int = 10
    extraction_sample_tokens_per_paper: int = 6000

    # Paper investigation tool limits.
    investigation_max_tokens: int = 6000
    investigation_max_entities: int = 100

    # Validation snippet window size.
    validation_snippet_tokens: int = 4096


@dataclass
class StarlingConfig:
    mode: StarlingMode = StarlingMode.run
    run_name: str = ""

    task: str | None = None
    spec_file: str | None = None

    models: StarlingModelsConfig = field(default_factory=StarlingModelsConfig)
    outputs: StarlingOutputsConfig = field(default_factory=StarlingOutputsConfig)
    extract: StarlingExtractionConfig = field(default_factory=StarlingExtractionConfig)
    tuning: StarlingAgentTuningConfig = field(default_factory=StarlingAgentTuningConfig)


def register_starling_config() -> None:
    slug = coolname.generate_slug(2)
    OmegaConf.register_new_resolver("coolname", lambda: slug)
    cs = ConfigStore.instance()
    cs.store(name="starling", node=StarlingConfig)
