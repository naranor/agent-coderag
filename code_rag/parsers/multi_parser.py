import os
import logging
from typing import List, Dict
from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit
from .ast_index import AstIndexParser

logger = logging.getLogger(__name__)

class MultiParser(IParser):
    """
    Delegates parsing to specific parsers based on file extension.
    """
    
    def __init__(self) -> None:
        self.parsers: Dict[str, IParser] = {
            ".py": AstIndexParser()
        }
        
        # Try to load JavaParser if javalang is available
        try:
            from .java_parser import JavaParser  # pylint: disable=import-outside-toplevel
            self.parsers[".java"] = JavaParser()
            logger.info("Java support enabled.")
        except ImportError:
            logger.warning("javalang not found. Java support disabled.")

    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        """
        Parses a file using the appropriate parser for its extension.
        """
        _, ext = os.path.splitext(file_path)
        parser = self.parsers.get(ext)
        
        if not parser:
            logger.debug("No parser for extension %s", ext)
            return []
            
        return await parser.distill_file(file_path)
