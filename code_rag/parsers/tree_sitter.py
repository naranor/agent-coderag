import hashlib
import logging
import importlib
from typing import List, Optional, Dict, Any
from pathlib import Path

import aiofiles
from tree_sitter import Language, Parser, Node

from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit, UnitKind
from .languages import LanguageConfig, LANGUAGES

logger = logging.getLogger(__name__)


class GrammarNotFoundError(Exception):
    """Raised when a tree-sitter grammar package is missing."""


class TreeSitterParser(IParser):
    """
    Asynchronous AST parser using Tree-Sitter for high-precision symbol extraction.
    """

    def __init__(self) -> None:
        self.parsers: Dict[str, Parser] = {}

    def _get_parser(self, lang_config: LanguageConfig) -> Parser:
        lang_id = lang_config.id
        if lang_id in self.parsers:
            return self.parsers[lang_id]

        try:
            # Dynamically load grammar (e.g. tree_sitter_python)
            module_name = lang_config.package
            lang_module = importlib.import_module(module_name)
            language = Language(lang_module.language())
            parser = Parser(language)
            self.parsers[lang_id] = parser
            return parser
        except ImportError as e:
            logger.error("Tree-sitter grammar for '%s' not installed.", lang_id)
            raise GrammarNotFoundError(
                f"[MISSING DEPENDENCY] Please install: pip install {lang_config.package}"
            ) from e

    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        """
        Parses a file and extracts high-level units (classes, functions).
        """
        ext = Path(file_path).suffix.lower()
        lang_config = next((cfg for cfg in LANGUAGES if ext in cfg.extensions), None)

        if not lang_config:
            logger.debug("No language config for extension %s", ext)
            return []

        try:
            async with aiofiles.open(file_path, mode="rb") as f:
                source = await f.read()

            parser = self._get_parser(lang_config)
            tree = parser.parse(source)

            units: List[KnowledgeUnit] = []
            ctx: Dict[str, Any] = {
                "source": source,
                "config": lang_config,
                "file_path": file_path,
                "units": units,
                "scope": None,
            }
            self._recursive_distill(tree.root_node, ctx)
            return units
        except GrammarNotFoundError:
            # Re-raise grammar errors so callers can handle them (e.g. CLI setup suggestion)
            raise
        except Exception as e:
            logger.error("Failed to parse %s: %s", file_path, e)
            return []

    def _recursive_distill(self, node: Node, ctx: Dict[str, Any]) -> None:
        config = ctx["config"]
        current_scope = ctx["scope"]

        if node.type in config.canonical_map:
            self._process_node(node, ctx)
            if config.canonical_map.get(node.type) == "CLASS":
                node_name = self._resolve_name(node)
                scope = ctx["scope"]
                current_scope = f"{scope}.{node_name}" if scope else node_name

        # Recurse into children with updated scope
        child_ctx = {**ctx, "scope": current_scope}
        for child in node.children:
            self._recursive_distill(child, child_ctx)

    def _process_node(self, node: Node, ctx: Dict[str, Any]) -> None:
        """Processes a single node and adds it to the units list."""
        node_name = self._resolve_name(node)
        docstring = self._extract_docstring(node)
        signature = self._extract_signature(node, ctx["source"], ctx["config"])

        # FULL code for distillation
        raw_text = node.text.decode("utf-8", errors="replace") if node.text else ""
        code_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        kind = self._determine_kind(node, ctx["config"])

        # Stable ID based on qualified name
        scope = ctx["scope"]
        qname = f"{scope}.{node_name}" if scope else node_name
        unit_id = f"{ctx['file_path']}:{qname}"

        unit = KnowledgeUnit(
            id=unit_id,
            name=node_name,
            kind=kind,
            signature=signature,
            docstring=docstring,
            path=ctx["file_path"],
            code_hash=code_hash,
            metadata={
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "node_type": node.type,
                "raw_code": raw_text,
            },
        )
        ctx["units"].append(unit)

    def _determine_kind(self, node: Node, config: LanguageConfig) -> UnitKind:
        raw_kind = config.canonical_map.get(node.type, "function").lower()
        try:
            return UnitKind(raw_kind)
        except ValueError:
            return UnitKind.FUNCTION

    def _resolve_name(self, node: Node) -> str:
        """Attempts to find a name, resolving anonymous blocks from context."""
        name_node = node.child_by_field_name("name")
        if not name_node:
            for child in node.children:
                if child.type in (
                    "identifier",
                    "property_identifier",
                    "type_identifier",
                ):
                    name_node = child
                    break

        if name_node and name_node.text:
            return name_node.text.decode("utf-8", errors="replace")

        parent = node.parent
        if parent:
            if parent.type == "variable_declarator":
                id_node = parent.child_by_field_name("name")
                if id_node and id_node.text:
                    return id_node.text.decode("utf-8", errors="replace")
            elif parent.type == "assignment":
                left_node = parent.child_by_field_name("left")
                if left_node and left_node.text:
                    return left_node.text.decode("utf-8", errors="replace")

        return "anonymous"

    def _extract_docstring(self, node: Node) -> Optional[str]:
        """Collects all preceding comment nodes as a docstring."""
        comments: List[str] = []
        curr = node.prev_sibling
        while curr and "comment" in curr.type:
            if curr.text:
                comment_text = curr.text.decode("utf-8", errors="replace").strip()
                clean_text = self._clean_comment(comment_text)
                if clean_text:
                    comments.insert(0, clean_text)
            curr = curr.prev_sibling

        return "\n".join(comments) if comments else None

    def _clean_comment(self, text: str) -> str:
        """Basic cleanup of common comment markers."""
        if text.startswith("/*"):
            text = text[2:]
        if text.endswith("*/"):
            text = text[:-2]
        lines = text.splitlines()
        clean_lines = []
        for line in lines:
            line = line.strip()
            for marker in ("///", "//", "/**", "*", "#"):
                if line.startswith(marker):
                    line = line[len(marker) :].strip()
                    break
            if line:
                clean_lines.append(line)
        return "\n".join(clean_lines)

    def _extract_signature(
        self, node: Node, source: bytes, config: LanguageConfig
    ) -> str:
        body_node = self._find_body(node, config)
        if not body_node:
            return node.text.decode("utf-8", errors="replace") if node.text else ""

        start = node.start_byte
        body_start = body_node.start_byte
        body_end = body_node.end_byte
        end = node.end_byte

        prefix = source[start:body_start].decode("utf-8", errors="replace")
        suffix = source[body_end:end].decode("utf-8", errors="replace")
        return prefix + config.stub_suffix + suffix

    def _find_body(self, node: Node, config: LanguageConfig) -> Optional[Node]:
        for field_name in config.body_fields:
            body = node.child_by_field_name(field_name)
            if body:
                return body
        for child in node.children:
            if child.type in config.fallback_bodies:
                return child
        return None
