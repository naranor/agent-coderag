import re
import logging
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit, UnitKind

logger = logging.getLogger(__name__)

class AstIndexParser(IParser):
    """
    Parser that uses ast-index CLI to discover code structure.
    Uses textual output parsing for maximum stability.
    """
    
    def __init__(self, binary_path: str = "ast-index"):
        self.binary_path = binary_path

    def _calculate_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def parse_file(self, path: str) -> List[KnowledgeUnit]:
        """
        Runs 'ast-index outline' and parses the textual results.
        Example output line: '  :182 V4Engine [class]'
        """
        try:
            cmd = [self.binary_path, "outline", path]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.error(f"ast-index failed for {path}: {process.stderr}")
                return []

            # 1. Read file content for bodies and signatures
            try:
                with open(path, "r") as f:
                    file_content = f.read()
                lines = file_content.splitlines()
            except Exception as e:
                logger.error(f"Failed to read file {path}: {e}")
                return []

            units = []
            
            # 2. Parse textual outline
            # Regex to match: optional spaces, colon, line number, name, optional signature, [kind]
            # Pattern: \s*:(\d+)\s+([^\s\[]+)(.*?)\s*\[([^\]]+)\]
            pattern = re.compile(r"\s*:(\d+)\s+([^\s\[]+)(.*?)\s*\[([^\]]+)\]")
            
            for line in process.stdout.splitlines():
                match = pattern.search(line)
                if not match:
                    continue
                
                line_num = int(match.group(1))
                name = match.group(2)
                signature = match.group(3).strip()
                kind_str = match.group(4).lower()
                
                # Map kinds
                kind_map = {
                    "class": UnitKind.CLASS,
                    "function": UnitKind.FUNCTION,
                    "method": UnitKind.METHOD,
                    "module": UnitKind.MODULE
                }
                kind = kind_map.get(kind_str, UnitKind.FUNCTION)
                
                # Extract code body (heuristic: 50 lines or until EOF)
                # In a real system, we'd use line numbers of the NEXT symbol
                body = "\n".join(lines[line_num-1 : line_num+50])
                
                units.append(KnowledgeUnit(
                    id=f"{path}:{name}",
                    kind=kind,
                    name=name,
                    path=path,
                    signature=signature or None,
                    code_hash=self._calculate_hash(body),
                    metadata={"raw_code": body}
                ))
            
            # 3. Handle whole module if no symbols found
            if not units:
                units.append(KnowledgeUnit(
                    id=f"{path}:module",
                    kind=UnitKind.MODULE,
                    name=Path(path).name,
                    path=path,
                    code_hash=self._calculate_hash(file_content),
                    metadata={"raw_code": file_content}
                ))

            return units
        except Exception as e:
            logger.error(f"Error in AstIndexParser for {path}: {e}")
            return []

    async def get_relations(self, path: str) -> List[Any]:
        return []
