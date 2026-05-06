import logging
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_ai import ModelRetry

from starling.infra.entities import ENTITY_TYPES, EntityCNF
from starling.infra.normalizer_base import EntityNormalizer, NormMatch

logger = logging.getLogger(__name__)


class EntityException(ModelRetry):
    """Base exception for entity-related errors."""

    def __init__(self, message: str, hint: str) -> None:
        super().__init__(message)
        self.hint = hint

    def user_message(self) -> str:
        """Format a user-friendly message combining the error and hint."""
        message = str(self)
        if message.endswith((".", "!", "?")):
            return f"{message} {self.hint}"
        return f"{message}. {self.hint}"


class EntityFilterError(EntityException):
    """Raised when entity filter structure is invalid."""


class EntityNormalizationError(EntityException):
    """Raised when entity terms cannot be normalized."""

    def __init__(self, failed_terms: list[str]) -> None:
        self.failed_terms = failed_terms
        terms_str = ", ".join(f'"{t}"' for t in failed_terms)
        super().__init__(
            f"Could not normalize entity terms: {terms_str}",
            hint="Use `normalize_entities` to verify terms and adjust the filter.",
        )


class FilterScope(StrEnum):
    paper = "paper"
    paragraph = "paragraph"
    chunk = "chunk"


@dataclass
class ResolvedEntityGroup:
    """Result of resolving one CNF entity group.

    Each term can be positive or negated; groups are ORs across all terms.
    """

    entity_types: list[str] = field(default_factory=list)
    """Positive entity type names (e.g., 'Gene', 'Disease')."""

    global_ids: list[str] = field(default_factory=list)
    """Resolved positive global identifiers (including expanded children)."""

    neg_entity_types: list[str] = field(default_factory=list)
    """Negated entity type names (e.g., '~Disease')."""

    neg_global_id_terms: list[list[str]] = field(default_factory=list)
    """Per-term lists of negated global identifiers (including expanded children)."""

    failed_terms: list[str] = field(default_factory=list)
    """Terms that could not be normalized."""

    @property
    def has_terms(self) -> bool:
        """True if this group has any terms."""
        return bool(self.entity_types or self.global_ids or self.neg_entity_types or self.neg_global_id_terms)

    @property
    def has_positive_terms(self) -> bool:
        return bool(self.entity_types or self.global_ids)

    @property
    def has_negated_terms(self) -> bool:
        return bool(self.neg_entity_types or self.neg_global_id_terms)


async def resolve_entity_groups(
    entity_groups: EntityCNF,
    normalizer: EntityNormalizer,
    *,
    norm_top_k: int | None = None,
    norm_ont_top_k: int | None = 3,
    allow_partial: bool = False,
    expand_entities: bool = True,
    max_depth: int = 4,
) -> list[ResolvedEntityGroup]:
    r"""Resolve CNF entity groups to global identifiers.

    Args:
        entity_groups: CNF filter - outer list is AND, inner lists are OR.
            Each term can be an entity type name (e.g., "Gene") or a
            specific entity term (e.g., "aspirin"). Terms prefixed with "\~"
            are negated (e.g., "~diabetes" excludes diabetes matches). Mixed
            positive and negated terms within a group are allowed.
        normalizer: Entity normalizer for term resolution.
        norm_top_k: Number of normalization matches to consider.
        norm_ont_top_k: Number of ontology-specific matches to consider (each ontology gets up to this many).
        allow_partial: Allow word-level partial matching during normalization.
        expand_entities: Expand resolved entities to include narrower/child concepts.
        max_depth: Maximum depth for child concept expansion.

    Returns:
        List of ResolvedEntityGroup, one per input group.

    Raises:
        EntityFilterError: If entity_groups structure is invalid.

    Example:
        groups = await resolve_entity_groups(
            [["aspirin", "\~ibuprofen"], ["\~Disease"]],
            normalizer,
            max_depth=2,
        )
        # groups[0].global_ids contains resolved IDs for aspirin + children
        # groups[0].neg_global_id_terms contains resolved IDs for ibuprofen + children
        # groups[1].neg_entity_types contains ["Disease"]
    """
    # Validate structure before processing
    _validate_entity_groups_structure(entity_groups)

    # First pass: parse terms and split positive/negated
    # Each entry: (pos_entity_types, neg_entity_types, pos_terms, neg_terms)
    group_terms: list[tuple[list[str], list[str], list[str], list[str]]] = []
    all_terms_to_normalize: set[str] = set()

    for group in entity_groups:
        deduped_group = list(dict.fromkeys(group))
        pos_entity_types: list[str] = []
        neg_entity_types: list[str] = []
        pos_terms: list[str] = []
        neg_terms: list[str] = []

        for term in deduped_group:
            clean_term, negated = _parse_term(term)

            if clean_term in ENTITY_TYPES:
                if negated:
                    neg_entity_types.append(clean_term)
                else:
                    pos_entity_types.append(clean_term)
            else:
                all_terms_to_normalize.add(clean_term)
                if negated:
                    neg_terms.append(clean_term)
                else:
                    pos_terms.append(clean_term)

        group_terms.append((pos_entity_types, neg_entity_types, pos_terms, neg_terms))

    # Normalize all terms (without ~ prefix)
    norm_results: dict[str, list[NormMatch]] = {}
    if all_terms_to_normalize:
        norm_results = await normalizer.normalize_many(
            list(all_terms_to_normalize),
            top_k=norm_top_k,
            ont_top_k=norm_ont_top_k,
            allow_partial=allow_partial,
        )

    # Build term -> gids mapping
    gids_to_expand: list[str] = []
    term_to_gids: dict[str, list[str]] = {}

    for term in all_terms_to_normalize:
        matches = norm_results.get(term, [])
        if matches:
            gids = [m.global_identifier for m in matches]
            term_to_gids[term] = gids
            if expand_entities:
                gids_to_expand.extend(gids)

    children_map: dict[str, list[str]] = {}
    if gids_to_expand:
        children_map = await normalizer.get_narrower_ids_many(gids_to_expand, max_depth=max_depth)

    # Build resolved groups
    resolved_groups: list[ResolvedEntityGroup] = []
    for pos_entity_types, neg_entity_types, pos_terms, neg_terms in group_terms:
        pos_gid_terms, pos_failed = _collect_gids_by_term(
            pos_terms,
            term_to_gids,
            children_map,
        )
        neg_gid_terms, neg_failed = _collect_gids_by_term(
            neg_terms,
            term_to_gids,
            children_map,
            failure_prefix="~",
        )
        neg_gid_terms = _dedupe_gid_terms(neg_gid_terms)

        pos_global_ids = list(dict.fromkeys(gid for term in pos_gid_terms for gid in term))
        resolved_groups.append(
            ResolvedEntityGroup(
                entity_types=list(dict.fromkeys(pos_entity_types)),
                global_ids=pos_global_ids,
                neg_entity_types=list(dict.fromkeys(neg_entity_types)),
                neg_global_id_terms=neg_gid_terms,
                failed_terms=pos_failed + neg_failed,
            )
        )

    return resolved_groups


def extract_all_failed_terms(groups: list[ResolvedEntityGroup]) -> list[str]:
    """Collect all failed terms from resolved groups."""
    failed: list[str] = []
    for group in groups:
        failed.extend(group.failed_terms)
    return failed


def _validate_groups(groups: list[ResolvedEntityGroup]) -> None:
    """Validate resolved entity groups for entity-based search.

    This should be called AFTER resolve_entity_groups to validate the resolved output.
    Input validation happens in resolve_entity_groups via _validate_entity_groups_structure.

    Raises:
        EntityFilterError: If resolved groups contain no valid terms after normalization.
        EntityNormalizationError: If any entity terms failed to normalize.
        ValueError: If groups have invalid structure (mixed positive/negative, etc).
    """
    # groups should not be empty at this point (caught by input validation)
    # But we still check for safety
    if not groups:
        raise EntityFilterError(
            "No groups provided for validation",
            hint="Provide entity_groups parameter.",
        )

    # Check for normalization failures FIRST (before checking if groups are empty)
    # This ensures we report the root cause: terms failed to normalize
    failed_terms = extract_all_failed_terms(groups)
    if failed_terms:
        logger.error(
            f"Entity normalization failure: {len(failed_terms)} terms could not be resolved: {failed_terms[:10]}"
        )
        raise EntityNormalizationError(failed_terms)

    # Check if all groups resolved to nothing (all terms succeeded but none matched)
    if not any(g.has_terms for g in groups):
        raise EntityFilterError(
            "All entity terms succeeded but resolved to no valid groups",
            hint="This should not happen - contact support.",
        )

    # Validate group structure
    has_any_positive_group = False

    for group in groups:
        if group.has_positive_terms and group.has_negated_terms:
            raise EntityFilterError(
                message="Search does not support mixed groups "
                "(positive and negative terms in the same group). "
                "Use separate groups: [[pos_terms], [~neg_terms]] instead of [[pos, ~neg]].",
                hint="Separate positive and negative terms into different groups.",
            )
        if group.has_positive_terms:
            has_any_positive_group = True

    if not has_any_positive_group and any(g.has_negated_terms for g in groups):
        raise EntityFilterError(
            message="Search requires at least one positive group when using negation. Pure negative filters are not supported.",
            hint="Add at least one positive entity group to the filter.",
        )


def _validate_entity_groups_structure(entity_groups: EntityCNF) -> None:
    """Validate entity_groups has correct structure before resolution.

    Raises:
        EntityFilterError: If structure is invalid.
    """
    if entity_groups is None:
        raise EntityFilterError("entity_groups must not be None", hint="Provide a list of entity groups")

    if not isinstance(entity_groups, list):
        raise EntityFilterError(
            f"entity_groups must be list[list[str]], got {type(entity_groups).__name__}",
            hint="Provide a list of entity groups, e.g. [['diabetes'], ['metformin']]",
        )

    if not entity_groups:
        raise EntityFilterError("entity_groups must not be empty", hint="Provide at least one entity group")

    # Check each group
    for i, group in enumerate(entity_groups):
        if not isinstance(group, (list, tuple)):
            # Catch the case where someone passes ["diabetes", "metformin"] instead of [["diabetes"], ["metformin"]]
            if isinstance(group, str):
                raise EntityFilterError(
                    f"entity_groups[{i}] is a string ('{group}'), but should be a list of strings. "
                    f"Did you mean [['{group}']] instead of ['{group}']?",
                    hint="Each group must be a list of entity terms, e.g. [['diabetes'], ['metformin']]",
                )
            else:
                raise EntityFilterError(
                    f"entity_groups[{i}] must be a list, got {type(group).__name__}",
                    hint="Each group must be a list of entity terms",
                )

        if not group:
            raise EntityFilterError(
                f"entity_groups[{i}] is empty - all groups must contain at least one term",
                hint="Remove empty groups or add entity terms",
            )

        # Check each term in the group
        for j, term in enumerate(group):
            if not isinstance(term, str):
                raise EntityFilterError(
                    f"entity_groups[{i}][{j}] must be a string, got {type(term).__name__}: {term}",
                    hint="Entity terms must be strings",
                )

    # All-negative filter check: every group must not consist entirely of negated terms
    if all(all(term.startswith("~") for term in group) for group in entity_groups):
        raise EntityFilterError(
            "Entity filter contains only negated terms. At least one group must have a positive term.",
            hint="Add a positive entity term (e.g., an entity type like 'Gene') or remove the negation prefix '~'.",
        )


def _parse_term(term: str) -> tuple[str, bool]:
    """Parse a term, returning (clean_term, is_negated)."""
    if term.startswith("~"):
        return term[1:], True
    return term, False


def _collect_gids_by_term(
    terms: list[str],
    term_to_gids: dict[str, list[str]],
    children_map: dict[str, list[str]],
    *,
    failure_prefix: str = "",
) -> tuple[list[list[str]], list[str]]:
    """Resolve terms to per-term global IDs and expand with children."""
    gid_terms: list[list[str]] = []
    failed_terms: list[str] = []

    for term in terms:
        gids = term_to_gids.get(term, [])
        if gids:
            expanded: list[str] = []
            for gid in gids:
                expanded.append(gid)
                children = children_map.get(gid)
                if children:
                    expanded.extend(children)
            gid_terms.append(list(dict.fromkeys(expanded)))
        else:
            failed_terms.append(f"{failure_prefix}{term}")

    return gid_terms, failed_terms


def _dedupe_gid_terms(gid_terms: list[list[str]]) -> list[list[str]]:
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for gids in gid_terms:
        key = tuple(gids)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(gids)
    return deduped
