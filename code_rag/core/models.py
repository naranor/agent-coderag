from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class UnitKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class RelationType(str, Enum):
    IMPORTS = "imports"  # used
    CALLS = "calls"  # used
    INHERITS = "inherits"  # used
    DEFINES = "defines"  # used


class Relation(BaseModel):
    """Connection between knowledge units."""

    from_id: str  # used
    to_id: str  # used
    type: RelationType  # used


class KnowledgeUnit(BaseModel):
    """Represents a piece of code (function, class, module)."""

    id: str
    kind: UnitKind
    name: str
    path: str
    signature: Optional[str] = None
    summary: Optional[str] = None
    code_hash: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relations: List[Relation] = Field(default_factory=list)
