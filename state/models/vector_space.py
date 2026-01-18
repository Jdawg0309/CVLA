"""
Vector space graph models.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VectorSpace:
    id: str
    label: str
    matrices: Tuple[str, ...] = ()
    vectors: Tuple[str, ...] = ()
    subspaces: Tuple[str, ...] = ()
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class VectorSpaceRelationship:
    parent_id: str
    child_id: str
    relation: str
    metadata: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class VectorSpaceGraph:
    spaces: Tuple[VectorSpace, ...] = ()
    relationships: Tuple[VectorSpaceRelationship, ...] = ()
