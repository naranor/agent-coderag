# Vulture whitelist to suppress false positives in Pydantic models and Enums
import code_rag.core.models

code_rag.core.models.RelationType.IMPORTS
code_rag.core.models.RelationType.CALLS
code_rag.core.models.RelationType.INHERITS
code_rag.core.models.RelationType.DEFINES
code_rag.core.models.Relation.from_id
code_rag.core.models.Relation.to_id
code_rag.core.models.Relation.type
