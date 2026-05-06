from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

EXTRA_DETAILS_FIELD = "extra_details"
EXTRACTION_GUIDANCE_FILENAME = "extraction_guidance.json"


class SchemaFieldSpec(BaseModel):
    """A single field in an induced extraction schema."""

    name: str = Field(description="Snake_case field name")
    description: str = Field(default="", description="What this field means and how to extract it")
    value_type: str = Field(
        default="string",
        description="Expected JSON type for the value (string|number|boolean)",
    )
    allowed_values: list[str] = Field(default_factory=list, description="Closed set of allowed values; empty if open")

    @field_validator("allowed_values", mode="before")
    @classmethod
    def _coerce_none(cls, v: list[str] | None) -> list[str]:
        return v or []


class ExtractionGuidance(BaseModel):
    """Derived extraction guidance and schema for extractor models."""

    task: str | None = Field(
        default=None,
        description="Original user task, when available.",
    )
    task_instantiation: str = Field(
        default="",
        description="A concrete, unambiguous instantiation of the user task for extraction (used in extractor prompts).",
    )
    schema_spec: list[SchemaFieldSpec] = Field(
        default_factory=list,
        description="Derived schema for the `extracted` object in each ExtractionRecord.",
    )

    @classmethod
    def from_path(cls, path: str | Path) -> ExtractionGuidance:
        return cls.model_validate_json(Path(path).read_text())

    def write_to_dir(self, output_dir: str | Path) -> Path:
        path = Path(output_dir) / EXTRACTION_GUIDANCE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path


class ExtractionSupport(BaseModel):
    """Readable, source-grounded support for an extraction."""

    paragraph_idx: int = Field(description="Anchor paragraph index from the `[p:<idx>]` labels in the input.")
    support_text: str = Field(
        description=(
            "A coherent, self-contained rendering of the supporting paper text. Stay close to the source wording "
            "when reasonable, but it does not need to be verbatim."
        )
    )


class ExtractionRecord(BaseModel):
    """A single extracted data point."""

    pmid: str
    extraction_id: str | None = None
    support: ExtractionSupport
    global_identifier: str | None = Field(
        default=None,
        description=(
            "If the extraction refers to an entity (small molecule, gene, or protein) listed in the input's "
            "entity list, copy its `global_identifier` here. Otherwise null."
        ),
    )
    extracted: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    needs_more_context: bool = Field(
        default=False,
        description=(
            "True if the extraction is incomplete or uncertain because relevant information "
            "(e.g., a baseline value, full entity name, or method details) likely exists outside "
            "the provided text window."
        ),
    )


class PaperExtractionResult(BaseModel):
    """All extractions from one paper."""

    pmid: str
    title: str | None
    extractions: list[ExtractionRecord]
    status: Literal["success", "no_content", "failed"]
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ExtractionBatchResult(BaseModel):
    """Results from batch extraction."""

    task_instantiation: str
    results: list[PaperExtractionResult]
    total_papers: int
    successful_papers: int
    total_extractions: int
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    schema_spec: list[SchemaFieldSpec] | None = None

    @classmethod
    def empty(
        cls,
        *,
        task_instantiation: str,
        schema_spec: list[SchemaFieldSpec] | None = None,
    ) -> ExtractionBatchResult:
        return cls.from_results(
            task_instantiation=task_instantiation,
            results=[],
            total_papers=0,
            schema_spec=schema_spec,
        )

    @classmethod
    def from_results(
        cls,
        *,
        task_instantiation: str,
        results: list[PaperExtractionResult],
        total_papers: int | None = None,
        schema_spec: list[SchemaFieldSpec] | None = None,
    ) -> ExtractionBatchResult:
        return cls(
            task_instantiation=task_instantiation,
            results=results,
            total_papers=len(results) if total_papers is None else total_papers,
            successful_papers=sum(1 for result in results if result.status == "success"),
            total_extractions=sum(len(result.extractions) for result in results),
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            schema_spec=schema_spec,
        )

    @classmethod
    def load(cls, path: str | Path) -> ExtractionBatchResult:
        """Load from JSON or JSONL based on file extension."""
        p = Path(path)
        if p.suffix == ".json":
            return cls.model_validate_json(p.read_bytes())
        return cls.from_jsonl(p)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        guidance_path: str | Path | None = None,
        task_instantiation: str | None = None,
        schema_spec: list[SchemaFieldSpec] | None = None,
    ) -> ExtractionBatchResult:
        jsonl_path = Path(path)
        resolved_guidance_path = (
            Path(guidance_path) if guidance_path is not None else jsonl_path.with_name(EXTRACTION_GUIDANCE_FILENAME)
        )

        guidance: ExtractionGuidance | None = None
        if resolved_guidance_path.exists():
            guidance = ExtractionGuidance.from_path(resolved_guidance_path)

        results: list[PaperExtractionResult] = []
        with jsonl_path.open() as f:
            for line_no, line in enumerate(f, start=1):
                payload = line.strip()
                if not payload:
                    continue
                try:
                    results.append(PaperExtractionResult.model_validate_json(payload))
                except ValidationError as exc:
                    raise ValueError(f"Invalid JSONL record in {jsonl_path} at line {line_no}") from exc

        resolved_task_instantiation = (
            task_instantiation
            if task_instantiation is not None
            else (guidance.task_instantiation or None if guidance is not None else None)
        )
        assert resolved_task_instantiation is not None, (
            "Task instantiation must be provided either directly or via guidance."
        )

        resolved_schema_spec: list[SchemaFieldSpec] | None = (
            schema_spec if schema_spec is not None else (guidance.schema_spec if guidance else None)
        )

        return cls.from_results(
            task_instantiation=resolved_task_instantiation,
            results=results,
            schema_spec=resolved_schema_spec,
        )


# --- Extraction Sampling Types ---


@dataclass
class ExtractionSampleState:
    """Tracks accumulated extraction state across samples.

    This is used to measure whether extraction yield is declining (convergence signal).
    """

    # Counts exclude transient failures (status="failed")
    total_papers_sampled: int = 0
    total_papers_with_extractions: int = 0
    total_extractions: int = 0
    extraction_rate_history: list[float] = field(default_factory=list)
    extractions_per_sample: list[int] = field(default_factory=list)
    seen_identifiers: set[str] = field(default_factory=set)
    new_unique_per_sample: list[int] = field(default_factory=list)

    def add_sample(
        self,
        papers_attempted: int,
        papers_with_extractions: int,
        num_extractions: int,
        global_identifiers: list[str],
    ) -> int:
        """Add a sample result (excluding transient failures). Returns count of new unique identifiers."""
        self.total_papers_sampled += papers_attempted
        self.total_papers_with_extractions += papers_with_extractions
        self.total_extractions += num_extractions

        if papers_attempted > 0:
            self.extraction_rate_history.append(papers_with_extractions / papers_attempted)

        self.extractions_per_sample.append(num_extractions)

        new_unique = 0
        for gid in global_identifiers:
            if gid not in self.seen_identifiers:
                self.seen_identifiers.add(gid)
                new_unique += 1
        self.new_unique_per_sample.append(new_unique)

        return new_unique

    @property
    def overall_extraction_rate(self) -> float:
        if self.total_papers_sampled == 0:
            return 0.0
        return self.total_papers_with_extractions / self.total_papers_sampled

    @property
    def total_unique_identifiers(self) -> int:
        return len(self.seen_identifiers)

    @property
    def marginal_yield_declining(self) -> bool:
        """True if last 2+ samples show declining or stable new entity discovery."""
        if len(self.new_unique_per_sample) < 2:
            return False
        recent = self.new_unique_per_sample[-3:] if len(self.new_unique_per_sample) >= 3 else self.new_unique_per_sample
        return recent[-1] <= recent[0]


class ExtractionSampleRecord(BaseModel):
    """A single extraction record shown to the agent."""

    pmid: str
    extraction_id: str
    paragraph_idx: int
    support_text: str
    extracted: dict
    global_identifier: str | None = None
    confidence: float
    needs_more_context: bool = False


class ExtractionSampleResult(BaseModel):
    """Results from sampling extraction on filter matches."""

    papers_attempted: int = Field(description="Number of papers extraction was attempted on")
    papers_with_extractions: int = Field(description="Number of papers that yielded at least one extraction")
    papers_failed: int = Field(description="Number of papers where extraction failed")
    extraction_rate: float = Field(
        description="Fraction of non-failed papers that yielded extractions (excludes transient failures)"
    )

    total_extractions: int = Field(description="Total extraction records produced")
    avg_extractions_per_paper: float = Field(description="Average extractions per successful paper")

    sample_records: list[ExtractionSampleRecord] = Field(description="Sample of extraction records for review")
    failure_reasons: dict[str, int] = Field(description="Count of papers by failure reason")

    cumulative_papers_sampled: int = Field(
        description="Total non-failed papers sampled across all sample_extraction calls"
    )
    cumulative_extractions: int = Field(description="Total extractions across all samples")
    cumulative_extraction_rate: float = Field(description="Overall extraction rate across all non-failed samples")
    cumulative_unique_entities: int = Field(description="Unique entities discovered (by global_identifier)")
    new_unique_this_sample: int = Field(description="New unique entities discovered in this sample")
    marginal_yield_declining: bool = Field(description="True if new entity discovery is declining")


@dataclass
class ExtractorDeps:
    task: str
    pmid: str
    task_instantiation: str | None = None
    schema_spec: list[SchemaFieldSpec] | None = None
