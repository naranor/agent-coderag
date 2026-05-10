import importlib
import os
import hashlib
from typing import List, Optional

from tree_sitter import Language, Parser, Node

from ..core.interfaces import IParser
from ..core.models import KnowledgeUnit, UnitKind
from .languages import LANGUAGE_MAP, EXTENSION_TO_LANGUAGE, LanguageConfig


class GrammarNotFoundError(Exception):
    """Raised when a tree-sitter grammar package is not installed."""

    def __init__(self, ext: str, package: str):
        self.ext = ext
        self.package = package
        super().__init__(
            f"[MISSING DEPENDENCY] Grammar for '{ext}' files is not installed. "
            f"Action: Run 'pip install {package}' to enable analysis."
        )


class TreeSitterParser(IParser):
    """Universal parser using tree-sitter for multiple languages."""

    def __init__(self):
        self._parsers = {}

    def _get_parser(self, lang_id: str, ext: str) -> tuple[Parser, LanguageConfig]:
        config = LANGUAGE_MAP.get(lang_id)
        if not config:
            raise ValueError(f"No configuration for language: {lang_id}")

        if lang_id in self._parsers:
            return self._parsers[lang_id], config

        try:
            lang_module = importlib.import_module(config.package)
            # Dynamically call the language function from config
            lang_func = getattr(lang_module, config.language_function)
            ts_lang = Language(lang_func())

            parser = Parser(ts_lang)
            self._parsers[lang_id] = parser
            return parser, config
        except (ImportError, AttributeError) as exc:
            raise GrammarNotFoundError(ext, config.package) from exc

    async def distill_file(self, file_path: str) -> List[KnowledgeUnit]:
        ext = os.path.splitext(file_path)[1].lower()
        lang_id = EXTENSION_TO_LANGUAGE.get(ext)
        if not lang_id:
            return []

        parser, config = self._get_parser(lang_id, ext)

        with open(file_path, "rb") as f:
            source_bytes = f.read()

        tree = parser.parse(source_bytes)
        units: List[KnowledgeUnit] = []
        self._recursive_distill(
            tree.root_node, source_bytes, config, file_path, units, scope=""
        )
        return units

    # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    def _recursive_distill(
        self,
        node: Node,
        source: bytes,
        config: LanguageConfig,
        file_path: str,
        units: List[KnowledgeUnit],
        scope: str = "",
    ):
        current_scope = scope
        if node.type in config.entities:
            # For name, we try the 'name' field first, then search for identifiers
            name_node = node.child_by_field_name("name")
            if not name_node:
                # Fallback: search for common identifier types among children
                for child in node.children:
                    if child.type in (
                        "identifier",
                        "property_identifier",
                        "type_identifier",
                    ):
                        name_node = child
                        break

            # Safe access to text
            node_name = (
                name_node.text.decode("utf-8")
                if (name_node and name_node.text)
                else "anonymous"
            )

            # Update scope for children (e.g. methods inside this class)
            if config.canonical_map.get(node.type) == "CLASS":
                current_scope = f"{scope}.{node_name}" if scope else node_name

            # Extract signature (the stubbed version for RAG display)
            signature = self._extract_signature(node, source, config)

            # FULL code for distillation (important!)
            raw_text = node.text.decode("utf-8") if node.text else ""
            code_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

            raw_kind = config.canonical_map.get(node.type, "function").lower()
            try:
                kind = UnitKind(raw_kind)
            except ValueError:
                kind = UnitKind.FUNCTION

            # Stable ID based on qualified name
            qname = f"{scope}.{node_name}" if scope else node_name
            unit_id = f"{file_path}:{qname}"

            unit = KnowledgeUnit(
                id=unit_id,
                name=node_name,
                kind=kind,
                signature=signature,
                path=file_path,
                code_hash=code_hash,
                metadata={
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "node_type": node.type,
                    "raw_code": raw_text,  # Pass the WHOLE thing to the distiller
                },
            )
            units.append(unit)

        # Recurse into children
        for child in node.children:
            self._recursive_distill(
                child, source, config, file_path, units, scope=current_scope
            )

    def _extract_signature(
        self, node: Node, source: bytes, config: LanguageConfig
    ) -> str:
        body_node = self._find_body(node, config)
        if not body_node:
            # Safe access to text
            return node.text.decode("utf-8") if node.text else ""

        start = node.start_byte
        body_start = body_node.start_byte
        body_end = body_node.end_byte
        end = node.end_byte

        # Text before body + stub + text after body (e.g. closing brace)
        prefix = source[start:body_start].decode("utf-8")
        suffix = source[body_end:end].decode("utf-8")
        return prefix + config.stub_suffix + suffix

    def _find_body(self, node: Node, config: LanguageConfig) -> Optional[Node]:
        # Try named fields first
        for field_name in config.body_fields:
            body = node.child_by_field_name(field_name)
            if body:
                return body

        # Try fallback node types
        for child in node.children:
            if child.type in config.fallback_bodies:
                return child

        return None
