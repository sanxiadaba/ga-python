"""An evaluated genetic-algorithm individual."""

from dataclasses import dataclass
from typing import Generic, TypeVar

GeneT = TypeVar("GeneT")


@dataclass(frozen=True, slots=True)
class Individual(Generic[GeneT]):
    """Keep a gene and its fitness together as an immutable value."""

    gene: GeneT
    fitness: float
