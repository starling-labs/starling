from collections.abc import Sequence
from typing import Literal, get_args

EntityType = Literal[
    "Anatomy",
    "Assay/Result",
    "CellLine",
    "CellType",
    "ClinicalTrial",
    "Disease",
    "Gene",
    "GeneVariant",
    "GOTerm",
    "Organism",
    "Pathway",
    "Phenotype",
    "Protein",
    "Protein/GeneFamily",
    "SmallMolecule",
    "SmallMoleculeClass",
    "Antibody",
    "Peptide",
    "RNA",
]

ENTITY_TYPES: set[str] = set(get_args(EntityType))

# CNF filter types for entity-based paper filtering.
#
# A CNF filter is structured as: outer list = AND, inner lists = OR.
# Each term in a group can be either:
#   - An entity type name (e.g., "Gene", "Disease") to match any entity of that type
#   - A specific entity term (e.g., "aspirin", "BRCA1") that will be normalized to global IDs
#   - A negated term prefixed with "~" (e.g., "~diabetes") which negates that literal
#
# Negation rules:
#   - Terms prefixed with "~" negate that literal within its OR group.
#   - A group must be all-positive or all-negated; mixed groups are not allowed.
#   - At least one positive group is required when using negation (pure negative filters are rejected).
#   - All-negated groups (e.g., ["~A", "~B"]) mean "NOT A OR NOT B" = NOT (A AND B).
#
# Examples:
#   [["aspirin"]]                                            -> papers mentioning aspirin
#   [["aspirin"], ["diabetes"]]                              -> aspirin AND diabetes
#   [["aspirin", "ibuprofen"]]                               -> aspirin OR ibuprofen
#   [["Gene"], ["breast cancer"]]                            -> any Gene AND breast cancer
#   [["E. coli", "Klebsiella"], ["pig"], ["~Mus Musculus"]]  -> (E. coli OR Klebsiella) AND pig AND NOT Mus Musculus
#   [["aspirin"], ["~diabetes"]]                             -> aspirin AND NOT diabetes
#   [["Gene"], ["~BRCA1"], ["~BRCA2"]]                       -> any Gene AND NOT BRCA1 AND NOT BRCA2
#   [["Gene"], ["~BRCA1", "~BRCA2"]]                         -> any Gene AND (NOT BRCA1 OR NOT BRCA2)
#
# Invalid examples (rejected):
#   [["aspirin", "~diabetes"]]                               -> mixed positive/negative group
#   [["~SmallMolecule", "~blood brain barrier"]]             -> negative-only filter
#
# Current semantics around negation:
#   - For vector search where we are interacting with several paragraphs at once,
#     a negated entity CAN show up in the chunk as long as one of the paragraphs
#     passes the filter (i.e., a paragraph without the negated entity and with the
#     required positive entities). We may want to disallow it within the chunk
#     entirely.

EntityTerm = str | EntityType
"""A single term in a CNF group: entity name (e.g., 'aspirin'), type (e.g., 'Gene'), or negated (e.g., '~diabetes')."""

EntityGroup = Sequence[EntityTerm]
"""An OR group of entity terms. Any term in the group can match."""

EntityCNF = Sequence[EntityGroup]
"""CNF entity filter: outer = AND, inner = OR. All groups must have at least one match."""
