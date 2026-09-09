from code_rag.api.models import ApiReport


async def run_api(manager, library: str, *, lang: str | None = None) -> ApiReport:
    """Discovers and returns the public API report for a library."""
    language = lang or "python"
    report = await manager.discovery.extract_api(library, language=language)
    return ApiReport(library=library, language=language, report=report)
