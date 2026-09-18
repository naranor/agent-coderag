from code_rag.api.models import ApiReport
from code_rag.discovery.manager import DiscoveryManager


async def run_api(
    discovery: DiscoveryManager, library: str, *, lang: str | None = None
) -> ApiReport:
    """Discovers and returns the public API report for a library."""
    language = lang or "python"
    report = await discovery.extract_api(library, language=language)
    return ApiReport(library=library, language=language, report=report)
