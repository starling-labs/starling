from __future__ import annotations

import os
from typing import Literal

import fsspec
import fsspec.utils
from fsspec import AbstractFileSystem
from pydantic import BaseModel


class TaggedString(BaseModel):
    text: str
    idx: int


class Paragraph(BaseModel):
    text: TaggedString


class Figure(BaseModel):
    fig_id: str
    caption: TaggedString
    bbox: list[float] | None = None  # [x0, y0, x1, y1]
    page: int | None = None


class Table(BaseModel):
    table_id: str
    caption: TaggedString
    content: TaggedString
    page: int | None = None


SectionEntry = Paragraph | Figure | Table


class Section(BaseModel):
    title: TaggedString
    entries: list[SectionEntry | Section]
    body: bool  # Whether or not this section is considered part of the "body"


Section.model_rebuild()


class Paper(BaseModel):
    pmid: str
    abstract: list[Paragraph]
    sections: list[Section]
    source: Literal["PMC", "Dots"]

    def flat_paragraphs(self) -> list[Paragraph]:
        """Get all Paragraphs in the paper, in order."""
        paras: list[Paragraph] = []
        for p in self.abstract:
            paras.append(p)

        def _gather_section(sec: Section) -> None:
            for e in sec.entries:
                if isinstance(e, Paragraph):
                    paras.append(e)
                elif isinstance(e, Section):
                    _gather_section(e)

        for s in self.sections:
            _gather_section(s)

        return paras

    def flat_tagged_strings(self, *, only_body: bool = False) -> list[TaggedString]:
        """Get all TaggedStrings in the paper, in order.

        Args:
            only_body: If True, only include strings from body sections.
        """
        strings: list[TaggedString] = []
        for p in self.abstract:
            strings.append(p.text)

        def _gather_section(sec: Section) -> None:
            if only_body and not sec.body:
                return
            strings.append(sec.title)
            for e in sec.entries:
                if isinstance(e, Paragraph):
                    strings.append(e.text)
                elif isinstance(e, Figure):
                    strings.append(e.caption)
                elif isinstance(e, Table):
                    strings.append(e.caption)
                    strings.append(e.content)
                elif isinstance(e, Section):
                    _gather_section(e)

        for s in self.sections:
            _gather_section(s)

        return strings

    def to_markdown(self, only_body: bool = True) -> str:
        lines = [f"# {self.pmid}"]
        if self.abstract:
            lines.append("## Abstract")
            for p in self.abstract:
                lines.append(p.text.text)
                lines.append("")

        def _add_section(sec: Section, level: int) -> None:
            if only_body and not sec.body:
                return

            lines.append(f"{'#' * level} {sec.title.text}")
            lines.append("")
            for e in sec.entries:
                if isinstance(e, Paragraph):
                    lines.append(e.text.text)
                    lines.append("")
                elif isinstance(e, Figure):
                    lines.append(f"![Figure {e.fig_id}]({e.caption.text})")
                    lines.append("")
                elif isinstance(e, Table):
                    lines.append(f"**Table {e.table_id}**: {e.caption.text}")
                    lines.append("")
                    lines.append(e.content.text)
                    lines.append("")
                elif isinstance(e, Section):
                    _add_section(e, level + 1)

        for s in self.sections:
            _add_section(s, 2)

        return "\n".join(lines)

    def get_text_for_indices(self, indices: list[int] | set[int]) -> str:
        """Get concatenated text for specific chunk indices.

        Args:
            indices: Set of idx values to include.

        Returns:
            Matching text chunks joined by double newlines.
        """
        idx_set = set(indices)
        strings = self.flat_tagged_strings()
        matches = [s.text for s in strings if s.idx in idx_set]
        return "\n\n".join(matches)

    def get_paragraph_chunks(self, chunk_ids: list[int]) -> list[Paragraph]:
        paras = self.flat_paragraphs()
        id_set = set(chunk_ids)
        return [p for p in paras if p.text.idx in id_set]


def pmid_to_path(pmid: str | int) -> str:
    if isinstance(pmid, int):
        pmid = str(pmid)

    # Pad to 8 digits
    spmid = pmid.zfill(8)
    f1 = spmid[-2:]
    f2 = spmid[-4:-2]
    return f"{f1}/{f2}/{pmid}"


class PaperStore:
    """
    Paper storage accessor with configurable filesystem backend.

    Supports local paths, GCS (gs://), and S3 (s3://) via fsspec.

    For GCS with service account auth:
        store = PaperStore(root="gs://bucket/corpus", token="/path/to/service-account.json")

    For local storage:
        store = PaperStore(root="/path/to/corpus")

    Environment variables:
        STARLING_PAPER_ROOT: Default root path/URL for paper storage.
        GCP_SERVICE_ACCOUNT_FILE: Path to GCS service account JSON (used when root is gs://).
    """

    def __init__(
        self,
        root: str | None = None,
        *,
        token: str | dict | None = None,
    ) -> None:
        """
        Args:
            root: Base path/URL for paper storage. Defaults to STARLING_PAPER_ROOT env var.
            token: Authentication for cloud storage. For GCS, can be:
                - Path to service account JSON file
                - Dict with service account credentials
                - "google_default" for application default credentials
                - None to auto-detect from GCP_SERVICE_ACCOUNT_FILE for GCS paths
        """
        root_path = root or os.getenv("STARLING_PAPER_ROOT")
        if not root_path:
            raise ValueError(
                "Paper storage root not configured. Please set STARLING_PAPER_ROOT or pass a 'root' argument."
            )

        self.root = root_path.rstrip("/")
        self.fs: AbstractFileSystem

        if token is not None:
            # Explicit credentials provided
            protocol = fsspec.utils.get_protocol(self.root)
            self.fs = fsspec.filesystem(protocol, token=token)
        elif self.root.startswith("gs://"):
            # GCS path - use service account from env if available
            sa_file = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
            if sa_file:
                self.fs = fsspec.filesystem("gcs", token=sa_file)
            else:
                self.fs, _ = fsspec.core.url_to_fs(self.root)
        else:
            # Auto-detect filesystem from URL
            self.fs, _ = fsspec.core.url_to_fs(self.root)

    def get(self, pmid: int | str) -> Paper:
        """Load a paper by PMID."""
        path = f"{self.root}/{pmid_to_path(pmid)}.json"
        with self.fs.open(path, "rb") as f:
            return Paper.model_validate_json(f.read())

    def exists(self, pmid: int | str) -> bool:
        """Check if a paper with the given PMID exists."""
        path = f"{self.root}/{pmid_to_path(pmid)}.json"
        return self.fs.exists(path)

    async def aget(self, pmid: int | str) -> Paper | None:
        """Load a paper by PMID asynchronously."""
        import asyncio

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self.get, pmid)
        except FileNotFoundError:
            return None

    async def aexists(self, pmid: int | str) -> bool:
        """Check if a paper with the given PMID exists asynchronously."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.exists, pmid)
