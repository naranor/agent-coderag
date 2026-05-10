import os
import logging
from typing import List
from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit
from .tree_sitter import TreeSitterParser
from .languages import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)


class MultiParser(IParser):
    """
    Delegates parsing to TreeSitterParser for all supported languages.
    """

    def __init__(self) -> None:
        self.tree_sitter_parser = TreeSitterParser()

    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        """
        Parses a file using TreeSitterParser if the extension is supported.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in EXTENSION_TO_LANGUAGE:
            return await self.tree_sitter_parser.distill_file(file_path)

        logger.debug("No parser for extension %s", ext)
        return []
