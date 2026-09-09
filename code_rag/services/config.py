from code_rag.intelligence.distiller import DistillerConfig


def load_or_update_config(
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> DistillerConfig:
    """Loads the persisted config, optionally updating and saving it."""
    config = DistillerConfig.load()
    if not any([url, key, model, provider]):
        return config

    if url is not None:
        config.api_base = url
    if key is not None:
        config.api_key = key
    if model is not None:
        config.model = model
    if provider is not None:
        config.provider = provider

    config.save()
    return config
