"""Entity normalization tools for validating entity terms before use in filters."""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Tool

from starling.infra.entities import ENTITY_TYPES, EntityType
from starling.infra.normalizer_base import EntityNormalizer, NormMatch


class NormalizationResult(BaseModel):
    """Result of normalizing a single entity term."""

    input_term: str = Field(description="The original term that was normalized")
    matched: bool = Field(description="Whether the term was successfully matched")
    is_entity_type: bool = Field(default=False, description="Whether the term is a recognized entity type")
    matches: list[NormMatch] = Field(default_factory=list, description="All normalization matches found")


@dataclass
class NormalizeEntitiesTool:
    """Callable tool for entity normalization with bound normalizer."""

    normalizer: EntityNormalizer

    async def __call__(self, terms: list[str | EntityType]) -> list[NormalizationResult]:
        """Normalize entity terms to verify they can be used as filters.

        Use this tool before calling search tools to confirm your entity terms will be recognized.
        Results include input_term, matched, is_entity_type, and matches. For matched terms, the
        top hit is matches[0]. matched=False means unrecognized, is_entity_type=True means a valid
        entity type like "SmallMolecule".

        Args:
            terms: List of entity terms to normalize (e.g., ["aspirin", "breast cancer", "BRCA1"]).

        Returns:
            List of normalization results in the same order as input terms.
        """
        terms_to_normalize = [t for t in terms if t not in ENTITY_TYPES]
        norm_results = (
            await self.normalizer.normalize_many(terms_to_normalize, allow_partial=False) if terms_to_normalize else {}
        )

        results: list[NormalizationResult] = []
        for term in terms:
            if term in ENTITY_TYPES:
                results.append(NormalizationResult(input_term=term, matched=True, is_entity_type=True))
                continue

            matches = norm_results.get(term, [])
            results.append(
                NormalizationResult(
                    input_term=term,
                    matched=bool(matches),
                    matches=matches,
                )
            )

        return results


def normalize_entities_tool(normalizer: EntityNormalizer, allow_partial: bool = False) -> Tool:
    """Create a normalize_entities tool with the normalizer bound.

    Args:
        normalizer: The entity normalizer to use for lookups.
        allow_partial: Currently always disabled inside the tool — kept for call-site compatibility.
    """
    del allow_partial
    return Tool(NormalizeEntitiesTool(normalizer).__call__, name="normalize_entities")
