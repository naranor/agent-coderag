import ast
import hashlib
import os
import logging
from typing import List, Optional
from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit, UnitKind

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
            units = []
            
            # Module level unit
            file_hash = hashlib.sha256(source.strip().encode()).hexdigest()
            units.append(KnowledgeUnit(
                id=f"{file_path}:module",
                kind=UnitKind.MODULE,
                name=os.path.basename(file_path),
                path=file_path,
                code_hash=file_hash,
                metadata={"raw_code": source}
            ))

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    unit = self._parse_node(node, file_path, source)
                    if unit:
                        units.append(unit)
            
            return units
        except Exception as e:
            logger.error("Failed to parse %s: %s", file_path, e)
            return []

    def _parse_node(self, node, file_path: str, source: str) -> Optional[KnowledgeUnit]:
        """Helper to create a KnowledgeUnit from an AST node."""
        kind = UnitKind.FUNCTION
        if isinstance(node, ast.ClassDef):
            kind = UnitKind.CLASS
        elif hasattr(node, "parent") and isinstance(node.parent, ast.ClassDef):
            kind = UnitKind.METHOD

        # Extract source code for the node
        try:
            node_source = ast.get_source_segment(source, node) or ""
            node_hash = hashlib.sha256(node_source.strip().encode()).hexdigest()
            
            # Simplified signature extraction
            signature = None
            if hasattr(node, "args"):
                args = [arg.arg for arg in node.args.args]
                signature = f"({', '.join(args)})"

            return KnowledgeUnit(
                id=f"{file_path}:{node.name}",
                kind=kind,
                name=node.name,
                path=file_path,
                signature=signature,
                code_hash=node_hash,
                metadata={"raw_code": node_source}
            )
        except Exception:
            return None
