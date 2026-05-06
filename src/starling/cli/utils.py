from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clickhouse_connect.driver.exceptions import DatabaseError
from pydantic_ai import ModelRetry, ModelSettings
from pydantic_ai.capabilities import AbstractCapability

from starling.cli.extraction_types import (
    EXTRACTION_GUIDANCE_FILENAME,
    ExtractionGuidance,
    PaperExtractionResult,
    SchemaFieldSpec,
)
from starling.infra.entity_filter import EntityException
from starling.infra.papers import PaperStore, TaggedString
from starling.infra.vector_search_base import VectorSearchBase

PMIDS_TO_PROCESS_FILENAME = "pmids_to_process.json"


@dataclass
class Gemma4Thinking(AbstractCapability[Any]):
    def get_model_settings(self) -> ModelSettings | None:
        return ModelSettings(extra_body={"chat_template_kwargs": {"enable_thinking": True}})


@dataclass(frozen=True)
class ExtractionResumeState:
    resume_dir: Path
    output_path: Path
    pmids_to_process: list[str]
    completed_pmids: set[str]
    remaining_pmids: list[str]
    guidance: ExtractionGuidance


def handle_db_errors[T](
    func: Callable[..., Coroutine[Any, Any, T]],
) -> Callable[..., Coroutine[Any, Any, T]]:
    """Decorator that converts DB errors to ModelRetry for pydantic-ai tools."""

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return await func(*args, **kwargs)
        except EntityException as e:
            raise ModelRetry(e.user_message()) from e
        except (TimeoutError, DatabaseError) as e:
            raise ModelRetry(
                "Query timed out or failed. Try a more specific filter (fewer entity alternatives, "
                "more restrictive terms) or use semantic_query to narrow results first."
            ) from e

    return wrapper


def write_json_atomic(path: str, payload: dict) -> None:
    opath = Path(path)
    opath.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)

    os.replace(tmp_path, path)


def _reset_jsonl_output(path: str | Path) -> None:
    opath = Path(path)
    opath.parent.mkdir(parents=True, exist_ok=True)
    opath.write_text("")


def _append_jsonl(path: str | Path, payload: dict) -> None:
    with Path(path).open("a") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


def _validate_jsonl_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix != ".jsonl":
        raise ValueError(f"Checkpoint output path must end with .jsonl: {output_path}")
    return output_path


def load_pmids_from_json(path: str) -> list[str]:
    """Load and normalize PMIDs from a JSON array."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError(f"PMID JSON must be a JSON array of PMIDs: {path}")

    pmids: list[str] = []
    seen: set[str] = set()
    for value in payload:
        if value is None:
            continue
        pmid = str(value).strip()
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        pmids.append(pmid)
    return pmids


def load_completed_pmids_from_jsonl(path: str | Path) -> set[str]:
    """Load PMIDs already present in an extraction JSONL file.

    Tolerates a truncated last line (e.g. from an interrupted write).
    """
    output_path = Path(path)
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    last_bad_line: int | None = None

    with output_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                completed.add(json.loads(payload)["pmid"])
                last_bad_line = None
            except (json.JSONDecodeError, KeyError, TypeError):
                if last_bad_line is not None:
                    raise ValueError(
                        f"Invalid extraction JSONL record in {output_path} at line {last_bad_line}"
                    ) from None
                last_bad_line = line_no

    if last_bad_line is not None:
        print(f"Warning: skipped truncated last line in {output_path} (line {last_bad_line})")

    return completed


def load_extraction_resume_state(resume_dir: str | Path) -> ExtractionResumeState:
    """Load the artifacts needed to resume an interrupted extraction run."""
    resolved_resume_dir = Path(resume_dir)
    pmids_path = resolved_resume_dir / PMIDS_TO_PROCESS_FILENAME
    guidance_path = resolved_resume_dir / EXTRACTION_GUIDANCE_FILENAME
    output_path = resolved_resume_dir / "extractions.jsonl"

    if not resolved_resume_dir.exists():
        raise ValueError(f"Resume directory does not exist: {resolved_resume_dir}")
    if not resolved_resume_dir.is_dir():
        raise ValueError(f"Resume path is not a directory: {resolved_resume_dir}")
    if not pmids_path.exists():
        raise ValueError(f"Resume directory is missing {PMIDS_TO_PROCESS_FILENAME}: {pmids_path}")
    if not guidance_path.exists():
        raise ValueError(f"Resume directory is missing {EXTRACTION_GUIDANCE_FILENAME}: {guidance_path}")

    pmids_to_process = load_pmids_from_json(str(pmids_path))
    guidance = ExtractionGuidance.from_path(guidance_path)
    completed_pmids = load_completed_pmids_from_jsonl(output_path)
    remaining_pmids = [pmid for pmid in pmids_to_process if pmid not in completed_pmids]

    return ExtractionResumeState(
        resume_dir=resolved_resume_dir,
        output_path=output_path,
        pmids_to_process=pmids_to_process,
        completed_pmids=completed_pmids,
        remaining_pmids=remaining_pmids,
        guidance=guidance,
    )


class ExtractionOutputWriter:
    def __init__(self, output_path: str | Path):
        self.output_path = _validate_jsonl_output_path(output_path)

    def write_pmids_to_process(self, pmids: list[str]) -> None:
        output_path = self.output_path.parent / PMIDS_TO_PROCESS_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(pmids, indent=2) + "\n")
        print(f"Wrote {len(pmids)} PMIDs to {output_path}")

    def write_guidance(
        self,
        *,
        task: str | None = None,
        task_instantiation: str,
        schema_spec: list[SchemaFieldSpec] | None = None,
    ) -> None:
        guidance_path = ExtractionGuidance(
            task=task,
            task_instantiation=task_instantiation,
            schema_spec=schema_spec or [],
        ).write_to_dir(self.output_path.parent)
        print(f"Wrote extraction guidance to {guidance_path}")

    def build_result_callback(self, *, append: bool = False) -> Callable[[PaperExtractionResult], None]:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            print(f"Appending extraction results incrementally to {self.output_path}")
        else:
            _reset_jsonl_output(self.output_path)
            print(f"Writing extraction results incrementally to {self.output_path}")

        def _on_result(result: PaperExtractionResult) -> None:
            with self.output_path.open("a") as f:
                f.write(result.model_dump_json())
                f.write("\n")

        return _on_result


_PRUNE_KEYS = frozenset(
    {
        "kind",
        "run_id",
        "metadata",
        "timestamp",
        "usage",
        "id",
        "signature",
        "tool_call_id",
        "part_kind",
        "finish_reason",
    }
)
_PRUNE_PREFIXES = ("provider",)


def prune_obj(ob: dict | list | Any) -> dict | list | Any:
    """Recursively prune unneeded fields from a dict."""
    if isinstance(ob, list):
        return [prune_obj(item) for item in ob]
    if isinstance(ob, dict):
        odict = {}
        for k, v in ob.items():
            if k in _PRUNE_KEYS or k.startswith(_PRUNE_PREFIXES):
                continue
            odict[k] = "<redacted>" if k == "abstract_snippet" else prune_obj(v)
        return odict
    return ob


# --- Paper text formatting & windowing ---


_ENTITY_TYPE_PREFIX: dict[str, str] = {
    "SmallMolecule": "SM",
    "SmallMoleculeClass": "SM",
    "Gene": "Gene",
    "Protein": "Protein",
}


def _format_entity_hit(row: dict[str, object]) -> str:
    entity_type = str(row.get("entity_type") or "")
    prefix = _ENTITY_TYPE_PREFIX.get(entity_type, entity_type or "Entity")
    name = str(row.get("name") or "").strip()
    gid = str(row.get("global_identifier") or "").strip()
    surface = list(row.get("surface") or [])  # ty:ignore[invalid-argument-type]
    synonyms = list(row.get("alt") or [])  # ty:ignore[invalid-argument-type]
    return (
        f"  - {prefix}: Name: {name} | Surface forms: {surface!r} | Synonyms: {synonyms!r} | global_identifier: {gid}"
    )


def format_tagged_strings(
    strings: list[TaggedString],
    *,
    entity_hits: dict[int, list[dict[str, object]]] | None = None,
) -> str:
    """Format a list of TaggedStrings as ``[p:idx] text`` blocks.

    Optionally appends a deduplicated entity list (by ``global_identifier``) at the end.
    """
    hits = entity_hits or {}
    blocks: list[str] = []
    for ts in strings:
        text = ts.text.strip()
        if not text:
            continue
        blocks.append(f"[p:{ts.idx}] {text}")

    # Collect unique entities across all paragraphs (dedupe by global_identifier)
    seen_gids: set[str] = set()
    unique_entity_lines: list[str] = []
    for rows in hits.values():
        for row in rows:
            gid = str(row.get("global_identifier") or "").strip()
            if not gid or gid in seen_gids:
                continue
            seen_gids.add(gid)
            unique_entity_lines.append(_format_entity_hit(row))

    result = "\n\n".join(blocks)
    if unique_entity_lines:
        result += "\n\n## Entities\n" + "\n".join(unique_entity_lines)
    return result


async def fetch_entity_hits(
    searcher: VectorSearchBase | None,
    *,
    pmid: str,
    paragraph_idxs: set[int],
) -> dict[int, list[dict[str, object]]]:
    """Fetch entity metadata (small molecules, genes, proteins) for the given paragraphs."""
    if searcher is None or not paragraph_idxs:
        return {}

    rows = await searcher.fetch_paragraph_entity_mentions([(pmid, idx) for idx in sorted(paragraph_idxs)])
    by_paragraph: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        paragraph = row.get("paragraph")
        if paragraph is None:
            continue
        by_paragraph.setdefault(int(paragraph), []).append(row)  # ty:ignore[invalid-argument-type]
    return by_paragraph


def iter_paragraph_windows(
    strings: list[TaggedString],
    paragraph_idx_set: set[int],
    window_paragraphs: int,
) -> list[tuple[int, int, list[TaggedString]]]:
    """Create contiguous windows of N paragraphs, including all TaggedStrings between them.

    The returned windows include:
    - All TaggedStrings between the first and last paragraph in the window (inclusive)
    - Adjacent non-paragraph TaggedStrings immediately before/after the window paragraphs

    Args:
        strings: TaggedStrings in document order (typically Paper.flat_tagged_strings()).
        paragraph_idx_set: Indices that correspond to paragraph strings.
        window_paragraphs: Number of paragraphs per window (must be >= 1).

    Returns:
        List of (start_paragraph_idx, end_paragraph_idx, window_strings).
    """
    if window_paragraphs <= 0:
        window_paragraphs = 1

    paragraph_positions = [pos for pos, ts in enumerate(strings) if ts.idx in paragraph_idx_set]
    if not paragraph_positions:
        return []

    windows: list[tuple[int, int, list[TaggedString]]] = []
    for start_i in range(0, len(paragraph_positions), window_paragraphs):
        end_i = min(start_i + window_paragraphs, len(paragraph_positions)) - 1
        start_pos = paragraph_positions[start_i]
        end_pos = paragraph_positions[end_i]

        # Include any non-paragraph strings directly adjacent to the paragraph span (titles, captions, tables, etc).
        while start_pos > 0 and strings[start_pos - 1].idx not in paragraph_idx_set:
            start_pos -= 1
        while end_pos + 1 < len(strings) and strings[end_pos + 1].idx not in paragraph_idx_set:
            end_pos += 1

        start_para_idx = strings[paragraph_positions[start_i]].idx
        end_para_idx = strings[paragraph_positions[end_i]].idx
        windows.append((start_para_idx, end_para_idx, list(strings[start_pos : end_pos + 1])))

    return windows


def format_with_context(
    strings: list[TaggedString],
    *,
    target_idxs: set[int] | None = None,
    context_radius: int = 5,
    entity_hits: dict[int, list[dict[str, object]]] | None = None,
) -> str | None:
    """Window tagged strings around target paragraphs and format as ``[p:idx]`` blocks.

    If ``target_idxs`` is provided, only includes paragraphs within
    ±``context_radius`` positions of each target. Otherwise formats all strings.
    """
    if not strings:
        return None

    if target_idxs:
        idx_to_pos = {ts.idx: i for i, ts in enumerate(strings)}
        n = len(strings)
        positions: set[int] = set()
        for idx in target_idxs:
            pos = idx_to_pos.get(idx)
            if pos is not None:
                positions.update(range(max(0, pos - context_radius), min(n, pos + context_radius + 1)))
        strings = [strings[i] for i in sorted(positions)]
        if not strings:
            return None

    # Filter entity hits to only include those in the returned strings
    if entity_hits:
        valid_idxs = {ts.idx for ts in strings}
        entity_hits = {idx: hits for idx, hits in (entity_hits or {}).items() if idx in valid_idxs}

    return format_tagged_strings(strings, entity_hits=entity_hits)


async def load_paper_context(
    paper_store: PaperStore,
    pmid: str,
    *,
    searcher: VectorSearchBase | None = None,
) -> tuple[list[TaggedString], dict[int, list[dict[str, object]]]] | None:
    """Load a paper's tagged strings and entity annotations.

    Returns ``(tagged_strings, entity_hits)`` or ``None`` if the paper is missing or empty.
    """
    paper = await paper_store.aget(pmid)
    if paper is None:
        return None

    strings = paper.flat_tagged_strings(only_body=True)
    if not strings:
        return None

    paragraph_idxs = {ts.idx for ts in strings}
    entity_hits = await fetch_entity_hits(searcher, pmid=pmid, paragraph_idxs=paragraph_idxs)
    return strings, entity_hits


async def format_paper_text(
    paper_store: PaperStore,
    pmid: str,
    *,
    target_idxs: set[int] | None = None,
    context_radius: int = 5,
    searcher: VectorSearchBase | None = None,
) -> str | None:
    """Fetch a paper and format its text as ``[p:idx]`` blocks with optional entity annotations.

    Convenience wrapper: loads paper context and delegates to :func:`format_with_context`.
    """
    ctx = await load_paper_context(paper_store, pmid, searcher=searcher)
    if ctx is None:
        return None
    strings, entity_hits = ctx
    return format_with_context(
        strings,
        target_idxs=target_idxs,
        context_radius=context_radius,
        entity_hits=entity_hits,
    )
