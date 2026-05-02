import javalang
import hashlib
import os
import logging
from typing import List, Any
from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit, UnitKind, Relation, RelationType

logger = logging.getLogger(__name__)

class JavaParser(IParser):
    """
    Parses Java code using javalang to extract classes, methods, and imports.
    """

    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        """
        Parses a Java file and returns a list of knowledge units.
        """
        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            lines = source.splitlines()
            tree = javalang.parse.parse(source)
            
            # File level unit (Module equivalent)
            file_hash = hashlib.sha256(source.strip().encode()).hexdigest()
            module_id = f"{file_path}:module"
            
            relations = []
            # Extract imports
            if hasattr(tree, 'imports'):
                for imp in tree.imports:
                    relations.append(Relation(
                        from_id=module_id,
                        to_id=imp.path,
                        type=RelationType.IMPORTS
                    ))

            units = [KnowledgeUnit(
                id=module_id,
                kind=UnitKind.MODULE,
                name=os.path.basename(file_path),
                path=file_path,
                code_hash=file_hash,
                metadata={"raw_code": source},
                relations=relations
            )]

            # Extract classes and methods
            for _, node in tree.filter(javalang.tree.ClassDeclaration):
                units.append(self._parse_class(node, file_path, lines))
                
                for method in node.methods:
                    units.append(self._parse_method(method, node.name, file_path, lines))
            
            # Extract interfaces
            for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
                units.append(self._parse_class(node, file_path, lines, is_interface=True))
                
                for method in node.methods:
                    if isinstance(method, javalang.tree.MethodDeclaration):
                        units.append(self._parse_method(method, node.name, file_path, lines))

            return [u for u in units if u is not None]
        except Exception as e:
            logger.error("Failed to parse Java file %s: %s", file_path, e)
            return []

    def _get_node_source(self, node: Any, lines: List[str]) -> str:
        """Heuristic to extract source code for a javalang node."""
        if not hasattr(node, 'position') or not node.position:
            return ""
        
        start_line = node.position.line - 1
        # javalang doesn't give end position easily. 
        # We take a reasonable chunk or look for the next node/closing brace.
        # For simplicity and accuracy in RAG, we'll take from start_line to a heuristic end.
        
        # Heuristic: find closing brace matching the first opening brace after start_line
        bracket_count = 0
        found_start = False
        captured_lines = []
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            captured_lines.append(line)
            for char in line:
                if char == '{':
                    bracket_count += 1
                    found_start = True
                elif char == '}':
                    bracket_count -= 1
            
            if found_start and bracket_count <= 0:
                break
        
        # If no braces found (like abstract methods), just return the single line
        if not found_start:
            return lines[start_line]
            
        return "\n".join(captured_lines)

    def _parse_class(self, node: Any, file_path: str, lines: List[str], is_interface: bool = False) -> KnowledgeUnit:
        """Helper to create a KnowledgeUnit from a Java class/interface."""
        node_source = self._get_node_source(node, lines)
        node_hash = hashlib.sha256(node_source.encode()).hexdigest()
        
        return KnowledgeUnit(
            id=f"{file_path}:{node.name}",
            kind=UnitKind.CLASS,
            name=node.name,
            path=file_path,
            code_hash=node_hash,
            metadata={
                "is_interface": is_interface,
                "raw_code": node_source
            }
        )

    def _parse_method(self, node: Any, class_name: str, file_path: str, lines: List[str]) -> KnowledgeUnit:
        """Helper to create a KnowledgeUnit from a Java method."""
        node_source = self._get_node_source(node, lines)
        
        params = [p.name for p in node.parameters]
        signature = f"({', '.join(params)})"
        
        # ID as path:ClassName.methodName
        unit_id = f"{file_path}:{class_name}.{node.name}"
        node_hash = hashlib.sha256(node_source.encode()).hexdigest()

        return KnowledgeUnit(
            id=unit_id,
            kind=UnitKind.METHOD,
            name=node.name,
            path=file_path,
            signature=signature,
            code_hash=node_hash,
            metadata={"raw_code": node_source}
        )
