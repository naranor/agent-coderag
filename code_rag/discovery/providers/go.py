import asyncio
import logging
import shutil
from .base import IDiscoveryProvider

logger = logging.getLogger(__name__)


class GoDiscoveryProvider(IDiscoveryProvider):
    """API discovery for Go packages using standard 'go doc'."""

    async def extract_api(self, library_name: str) -> str:
        """
        Extracts Go API using 'go doc -all <library_name>'.
        """
        go_bin = shutil.which("go")
        if not go_bin:
            return "Error: 'go' executable not found. Please ensure Go is installed and in PATH."

        logger.info("Running 'go doc -all %s'...", library_name)
        try:
            process = await asyncio.create_subprocess_exec(
                go_bin,
                "doc",
                "-all",
                library_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                err_msg = stderr.decode().strip()
                if "not found" in err_msg or "cannot find" in err_msg:
                    return (
                        f"Error: Go package '{library_name}' not found.\n"
                        "Action: If this is a third-party library, try running 'go mod download' in your project."
                    )
                return f"Error: 'go doc' failed: {err_msg}"

            output = stdout.decode().strip()
            if not output:
                return f"No documentation found for Go package '{library_name}'."

            return f"# Public API for Go Package '{library_name}':\n\n{output}"

        except Exception as e:
            return f"Failed to extract Go API: {e}"
