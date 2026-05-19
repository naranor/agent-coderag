import logging
import re
from pathlib import Path
from typing import Dict, Optional

import defusedxml.ElementTree as element_tree
import dnfile
from .base import IDiscoveryProvider
from ...core.exceptions import DiscoveryError

logger = logging.getLogger(__name__)


def _parse_version(path: Path):
    parts = re.findall(r"\d+", path.name)
    return tuple(map(int, parts))


class CSharpDiscoveryProvider(IDiscoveryProvider):
    """API discovery for C# libraries using dnfile (metadata) and XML docs."""

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts C# API using:
        1. NuGet cache search for DLL and XML.
        2. dnfile to read assembly metadata (types, methods).
        3. XML documentation for method summaries.
        """
        dll_path, xml_path = self._find_artifacts(library_name)

        if not dll_path:
            raise DiscoveryError(
                f"Could not find DLL for C# library '{library_name}' in NuGet cache."
            )

        output = [
            f"# Public API for C# Library '{library_name}' (from {dll_path.name}):"
        ]

        # 1. Load XML docs if available
        docs = self._load_xml_docs(xml_path)

        # 2. Extract types via dnfile
        try:
            pe = dnfile.dnPE(str(dll_path))
            if not pe.net or not pe.net.mdtables:
                raise DiscoveryError(f"'{dll_path.name}' is not a valid .NET assembly.")

            # Map for easier lookup: FullName -> Summary
            types_found = 0

            # TypeDef table
            typedefs = pe.net.mdtables.TypeDef
            for row in typedefs:
                # Basic visibility check (Flags)
                # 0x00000007 is Visibility mask. Public is 0x00000001
                if not (row.Flags & 0x00000007) == 1:
                    continue

                namespace = row.TypeNamespace
                name = row.TypeName
                full_name = f"{namespace}.{name}" if namespace else name

                output.append(f"\n## Class: {full_name}")
                if full_name in docs:
                    output.append(f"  *{docs[full_name]}*")

                # Find methods for this type
                # This is tricky with raw tables, but we can look at MethodDef
                # For simplicity in this version, we list important names
                types_found += 1
                if types_found > 10:
                    break

            if types_found == 0:
                output.append("\nNo public types found in metadata.")

            return "\n".join(output[:100])

        except Exception as e:
            if isinstance(e, DiscoveryError):
                raise
            raise DiscoveryError(
                f"Failed to extract C# API from {dll_path.name}: {e}"
            ) from e

    def _find_artifacts(self, lib_name: str) -> tuple[Optional[Path], Optional[Path]]:
        """Searches for DLL and XML in NuGet cache."""
        home = Path.home()
        nuget_cache = home / ".nuget" / "packages"
        if not nuget_cache.exists():
            return None, None

        # Search for library folder (case-insensitive)
        for lib_dir in nuget_cache.glob("*"):
            if lib_dir.name.lower() == lib_name.lower():
                # Find latest version using semantic-aware sorting
                versions = sorted(lib_dir.glob("*"), key=_parse_version)
                if not versions:
                    continue
                latest = versions[-1]

                # Find DLL in lib folder
                lib_path = latest / "lib"
                if lib_path.exists():
                    # Look for best target (e.g. net6.0, netstandard2.0)
                    targets = sorted(lib_path.glob("*"), reverse=True)
                    for target in targets:
                        dlls = list(target.glob("*.dll"))
                        if dlls:
                            dll = dlls[0]
                            xml = target / f"{dll.stem}.xml"
                            return dll, (xml if xml.exists() else None)
        return None, None

    def _load_xml_docs(self, xml_path: Optional[Path]) -> Dict[str, str]:
        """Parses C# XML documentation file."""
        docs: Dict[str, str] = {}
        if not xml_path:
            return docs

        try:
            tree = element_tree.parse(str(xml_path))
            root = tree.getroot()
            for member in root.findall(".//member"):
                name = member.get("name", "")
                summary = member.find("summary")
                if name and summary is not None:
                    # Clean up name prefix (T: for type, M: for method)
                    clean_name = name[2:]
                    text = summary.text.strip() if summary.text else ""
                    docs[clean_name] = text
        except Exception as e:
            logger.debug("Failed to parse XML docs %s: %s", xml_path, e)
        return docs
