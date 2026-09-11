from code_rag.core.exceptions import IntelligenceError
from code_rag.intelligence.distiller import DistillerConfig


def apply_config_updates(  # pylint: disable=too-many-arguments
    config: DistillerConfig,
    *,
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    embedding_url: str | None = None,
    embedding_key: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    clear_embedding: bool = False,
) -> DistillerConfig:
    if url is not None:
        config.api_base = url
    if key is not None:
        config.api_key = key
    if model is not None:
        config.model = model
    if provider is not None:
        config.provider = provider

    if clear_embedding:
        config.embedding_base = None
        config.embedding_key = None
        config.embedding_model = None
        config.embedding_provider = None

    if embedding_url is not None:
        config.embedding_base = embedding_url
    if embedding_key is not None:
        config.embedding_key = embedding_key
    if embedding_model is not None:
        config.embedding_model = embedding_model
    if embedding_provider is not None:
        config.embedding_provider = embedding_provider

    # Re-run validators after raw assignment (blank → None)
    if isinstance(config, DistillerConfig):
        config = DistillerConfig.model_validate(config.model_dump())

    base_set = config.embedding_base is not None
    model_set = config.embedding_model is not None
    if base_set != model_set:
        raise IntelligenceError(
            "embedding_base and embedding_model must both be set or both unset"
        )
    return config


def _has_updates(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    url,
    key,
    model,
    provider,
    embedding_url,
    embedding_key,
    embedding_model,
    embedding_provider,
    clear_embedding,
) -> bool:
    return any(
        [
            url is not None,
            key is not None,
            model is not None,
            provider is not None,
            embedding_url is not None,
            embedding_key is not None,
            embedding_model is not None,
            embedding_provider is not None,
            clear_embedding,
        ]
    )


def load_or_update_config(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    embedding_url: str | None = None,
    embedding_key: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    clear_embedding: bool = False,
) -> DistillerConfig:
    config = DistillerConfig.load()
    if not _has_updates(
        url,
        key,
        model,
        provider,
        embedding_url,
        embedding_key,
        embedding_model,
        embedding_provider,
        clear_embedding,
    ):
        return config
    config = apply_config_updates(
        config,
        url=url,
        key=key,
        model=model,
        provider=provider,
        embedding_url=embedding_url,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        clear_embedding=clear_embedding,
    )
    config.save()
    return config
