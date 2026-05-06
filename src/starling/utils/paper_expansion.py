from collections.abc import Sequence

import logfire
import tiktoken

from starling.infra.papers import (
    Figure,
    Paper,
    Paragraph,
    Section,
    SectionEntry,
    Table,
    TaggedString,
)
from starling.models.llm import get_tokenizer


def _token_counts(
    paper: Paper,
    tokenizer: tiktoken.Encoding,
    *,
    only_body: bool = False,
) -> tuple[list[TaggedString], list[int], dict[int, int], dict[int, int]]:
    strings = paper.flat_tagged_strings(only_body=only_body)

    encodings = tokenizer.encode_batch(
        [ts.text for ts in strings],
    )

    counts = [len(e) for e in encodings]

    idx_to_pos = {ts.idx: i for i, ts in enumerate(strings)}
    idx_to_tokens = {ts.idx: c for ts, c in zip(strings, counts)}

    return strings, counts, idx_to_pos, idx_to_tokens


def _get_nearest_valid_idx(strings: Sequence[TaggedString], center_idx: int) -> int | None:
    if not strings:
        return None

    pos_map = {ts.idx: i for i, ts in enumerate(strings)}
    if center_idx in pos_map:
        return center_idx

    nearest_idx = strings[0].idx
    for ts in strings:
        if abs(ts.idx - center_idx) < abs(nearest_idx - center_idx):
            nearest_idx = ts.idx

    return nearest_idx


def _select_contiguous_window(
    strings: Sequence[TaggedString], counts: Sequence[int], center_idx: int, total_budget: int
) -> set[int]:
    """
    Choose a contiguous window around `center_idx` whose total token count is <= total_budget. Start by allocating ~half budget to the left and ~half
    to the right (center is always included), then donate unused tokens to the other side greedily.
    """
    pos_map = {ts.idx: i for i, ts in enumerate(strings)}
    if center_idx not in pos_map:
        logfire.warning(f"TaggedString idx {center_idx} not found in Paper.")
        # set center idx to nearest valid idx
        center_idx = _get_nearest_valid_idx(strings, center_idx)  # ty:ignore[invalid-assignment]
        if center_idx is None:
            return set()

    if total_budget <= 0:
        return set()

    pos = pos_map[center_idx]
    n = len(strings)
    center_tokens = counts[pos]

    if center_tokens >= total_budget:
        return {strings[pos].idx}

    remaining = total_budget - center_tokens
    left_rem = remaining // 2
    right_rem = remaining - left_rem

    L = R = pos

    # First, consume from the left up to left_rem
    i = pos - 1
    while i >= 0 and counts[i] <= left_rem:
        L = i
        left_rem -= counts[i]
        i -= 1

    # Then, consume from the right up to right_rem
    j = pos + 1
    while j < n and counts[j] <= right_rem:
        R = j
        right_rem -= counts[j]
        j += 1

    # Donation phase
    leftover = left_rem + right_rem
    next_left = i  # the next candidate index on the left (not yet included)
    next_right = j  # the next candidate index on the right (not yet included)

    while True:
        can_left = (next_left >= 0) and (counts[next_left] <= leftover)
        can_right = (next_right < n) and (counts[next_right] <= leftover)

        if not can_left and not can_right:
            break

        if can_left and can_right:
            if counts[next_left] <= counts[next_right]:
                L = next_left
                leftover -= counts[next_left]
                next_left -= 1
            else:
                R = next_right
                leftover -= counts[next_right]
                next_right += 1
        elif can_left:
            L = next_left
            leftover -= counts[next_left]
            next_left -= 1
        else:  # can_right
            R = next_right
            leftover -= counts[next_right]
            next_right += 1

    return {ts.idx for ts in strings[L : R + 1]}


def _filter_by_tagged_idxs(
    paper: Paper,
    keep_idxs: set[int],
    *,
    drop_empty_sections: bool = True,
) -> Paper:
    def empty_ts() -> TaggedString:
        return TaggedString(text="", idx=-1)

    def filter_section(sec: Section) -> Section | None:
        new_entries: list[SectionEntry | Section] = []

        for e in sec.entries:
            if isinstance(e, Paragraph):
                if e.text.idx in keep_idxs:
                    new_entries.append(e)
            elif isinstance(e, Figure):
                if e.caption.idx in keep_idxs:
                    new_entries.append(e)
            elif isinstance(e, Table):
                keep_cap = e.caption.idx in keep_idxs
                keep_ctt = e.content.idx in keep_idxs
                if keep_cap or keep_ctt:
                    new_entries.append(
                        Table(
                            table_id=e.table_id,
                            caption=e.caption if keep_cap else empty_ts(),
                            content=e.content if keep_ctt else empty_ts(),
                            page=e.page,
                        )
                    )

            elif isinstance(e, Section):
                child = filter_section(e)
                if child is not None:
                    new_entries.append(child)

        title_kept = sec.title.idx in keep_idxs
        if title_kept:
            new_title = sec.title
        else:
            new_title = empty_ts()

        if (title_kept or new_entries) and not (
            drop_empty_sections and not title_kept and not new_entries and new_title.text == ""
        ):
            return Section(title=new_title, entries=new_entries, body=sec.body)

        return None

    new_abstract = [p for p in paper.abstract if p.text.idx in keep_idxs]

    new_sections: list[Section] = []
    for s in paper.sections:
        fs = filter_section(s)
        if fs is not None:
            new_sections.append(fs)

    return Paper(pmid=paper.pmid, abstract=new_abstract, sections=new_sections, source=paper.source)


def expand_around_idx(
    paper: Paper,
    center_idx: int,
    total_tokens: int,
    tokenizer: str | tiktoken.Encoding = "o200k_base",
    *,
    drop_empty_sections: bool = True,
    only_body: bool = True,
) -> Paper:
    """
    Slice the paper around `center_idx` into a new Paper containing a contiguous window that fits `total_tokens` tokens
    (approximately half before and half after, with donation of unused tokens).

    Args:
        paper: The paper to slice.
        center_idx: The TaggedString idx to center the window around.
        total_tokens: Maximum tokens for the window.
        tokenizer: Tokenizer name or instance.
        drop_empty_sections: If True, drop sections with no content.
        only_body: If True, only consider body sections when computing the window,
            avoiding token budget waste on non-body content that would be dropped anyway.

    Note that this is *not strict*. You may blow the model up if you push the limit.
    """
    if isinstance(tokenizer, str):
        tok = get_tokenizer(tokenizer)
    else:
        tok = tokenizer

    strings, counts, _, _ = _token_counts(paper, tok, only_body=only_body)
    keep = _select_contiguous_window(strings, counts, center_idx, total_tokens)
    return _filter_by_tagged_idxs(
        paper,
        keep,
        drop_empty_sections=drop_empty_sections,
    )
