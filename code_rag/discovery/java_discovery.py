import os
import subprocess  # nosec
import logging
import shutil
import re
from pathlib import Path
from typing import List, Optional
from ..core.utils import validate_path
from ..core.constants import DEFAULT_SUBPROCESS_TIMEOUT

logger = logging.getLogger(__name__)


def find_javap() -> Optional[str]:
    """Finds the javap executable."""
    return shutil.which("javap")


def _parse_version(path: Path):
    parts = re.findall(r"\d+", path.name)
    return tuple(map(int, parts))


def find_java_library_jar(library_name: str) -> List[Path]:
    """
    Attempts to find JAR files for a given library name in Maven and Gradle caches.
    Example: 'junit' or 'org.slf4j'
    """
    home = Path.home()
    search_paths = [
        home / ".m2" / "repository",
        home / ".gradle" / "caches" / "modules-2" / "files-2.1",
    ]

    found_jars = []
    # Replace dots with path separators for Maven style search
    lib_path_part = library_name.replace(".", os.sep)

    for base_path in search_paths:
        if not base_path.exists():
            continue

        # Maven style search
        maven_lib_dir = base_path / lib_path_part
        if maven_lib_dir.exists():
            for jar in maven_lib_dir.rglob("*.jar"):
                if "sources" not in jar.name and "javadoc" not in jar.name:
                    found_jars.append(jar)

        # Gradle style search (more complex due to hashes)
        if not found_jars and "gradle" in str(base_path):
            for group_dir in base_path.glob(f"**/{library_name}"):
                for jar in group_dir.rglob("*.jar"):
                    if "sources" not in jar.name and "javadoc" not in jar.name:
                        found_jars.append(jar)

    return found_jars


async def extract_java_api(library_name: str, jar_path: Optional[str] = None) -> str:
    """
    Extracts Java API using javap on found JAR files.
    """
    javap_bin = find_javap()
    if not javap_bin:
        return "Error: 'javap' not found. Please ensure JDK is installed and in PATH."

    if jar_path:
        target_jar = Path(jar_path)
    else:
        jars = find_java_library_jar(library_name)
        if not jars:
            return (
                f"Error: Could not find JAR files for library '{library_name}' "
                "in local Maven/Gradle caches."
            )
        # Take the most recent or highest version using semantic-aware sorting
        target_jar = sorted(jars, key=_parse_version)[-1]

    try:
        # 1. List classes in JAR
        jar_bin = shutil.which("jar")
        if not jar_bin:
            return f"Error: 'jar' utility not found. Cannot list classes in {target_jar.name}"

        # Validate target_jar path
        target_jar = validate_path(target_jar)

        result = subprocess.run(
            [jar_bin, "-tf", str(target_jar)],
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_SUBPROCESS_TIMEOUT,
        )  # nosec
        classes = [
            line.replace("/", ".").replace(".class", "")
            for line in result.stdout.splitlines()
            if line.endswith(".class") and "$" not in line
        ]  # Ignore inner classes

        if not classes:
            return f"No public classes found in {target_jar.name}"

        output = [
            f"# Public API for Java Library '{library_name}' (from {target_jar.name}):"
        ]

        # Limit to top 20 classes to avoid massive output
        for cls in classes[:20]:
            # Basic validation of class name string (prevent injection)
            if not all(c.isalnum() or c in "._$" for c in cls):
                continue

            res = subprocess.run(
                [javap_bin, "-public", "-classpath", str(target_jar), cls],
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_SUBPROCESS_TIMEOUT,
            )  # nosec
            if res.returncode == 0:
                # Basic cleaning of javap output
                lines = res.stdout.splitlines()
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("Compiled from")
                        or line.startswith("}")
                    ):
                        continue
                    if line.startswith("public class") or line.startswith(
                        "public interface"
                    ):
                        clean_lines.append(f"- **{line.replace('{', '')}**")
                    else:
                        clean_lines.append(f"  - `{line.replace(';', '')}`")
                output.extend(clean_lines)

        return "\n".join(output[:100])
    except Exception as e:
        return f"Failed to extract Java API: {e}"
