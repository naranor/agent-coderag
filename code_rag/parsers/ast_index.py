import ast
import hashlib
import os
import logging
from typing import List, Optional, Any
from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit, UnitKind, Relation, RelationType

logger = logging.getLogger(__name__)


class AstIndexParser(IParser):
    """
    Parses Python code using the built-in AST module to extract units.
    """

    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        """
        Parses a file and returns a list of knowledge units.
        """
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            # Module level unit
            file_hash = hashlib.sha256(source.strip().encode()).hexdigest()
            module_id = f"{file_path}:module"

            # Extract imports
            relations = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        relations.append(
                            Relation(
                                from_id=module_id,
                                to_id=alias.name,
                                type=RelationType.IMPORTS,
                            )
                        )
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    for alias in node.names:
                        full_name = (
                            f"{module_name}.{alias.name}" if module_name else alias.name
                        )
                        relations.append(
                            Relation(
                                from_id=module_id,
                                to_id=full_name,
                                type=RelationType.IMPORTS,
                            )
                        )

            units = [
                KnowledgeUnit(
                    id=module_id,
                    kind=UnitKind.MODULE,
                    name=os.path.basename(file_path),
                    path=file_path,
                    code_hash=file_hash,
                    metadata={"raw_code": source},
                    relations=relations,
                )
            ]

            # Recursive traversal to track parents and qualified names
            def traverse(
                node: ast.AST, parent_node: Optional[ast.AST] = None, scope: str = ""
            ):
                for child in ast.iter_child_nodes(node):
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        context = {"parent": parent_node, "scope": scope}
                        unit = self._parse_node(child, file_path, source, context)
                        if unit:
                            units.append(unit)
                            # Update scope for nested elements
                            new_scope = f"{scope}.{child.name}" if scope else child.name
                            traverse(child, child, new_scope)
                    else:
                        traverse(child, parent_node, scope)

            traverse(tree)
            return units
        except Exception as e:
            logger.error("Failed to parse %s: %s", file_path, e)
            return []

    def _parse_node(
        self, node: Any, file_path: str, source: str, context: Optional[dict] = None
    ) -> Optional[KnowledgeUnit]:
        """Helper to create a KnowledgeUnit from an AST node."""
        context = context or {}
        parent_node = context.get("parent")
        scope = context.get("scope", "")

        kind = UnitKind.FUNCTION
        if isinstance(node, ast.ClassDef):
            kind = UnitKind.CLASS
        elif isinstance(parent_node, ast.ClassDef):
            kind = UnitKind.METHOD

        # Extract source code for the node
        try:
            node_source = ast.get_source_segment(source, node) or ""
            node_hash = hashlib.sha256(node_source.strip().encode()).hexdigest()

            # Use qualified name for ID to prevent collisions (e.g., File:Class.Method)
            qname = f"{scope}.{node.name}" if scope else node.name
            unit_id = f"{file_path}:{qname}"

            # Simplified signature extraction
            signature = None
            if hasattr(node, "args"):
                args = [arg.arg for arg in node.args.args]
                signature = f"({', '.join(args)})"

            return KnowledgeUnit(
                id=unit_id,
                kind=kind,
                name=node.name,
                path=file_path,
                signature=signature,
                code_hash=node_hash,
                metadata={"raw_code": node_source},
            )
        except Exception:
            return None
