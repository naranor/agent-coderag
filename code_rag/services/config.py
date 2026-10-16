from code_rag.core.exceptions import IntelligenceError
from code_rag.intelligence.distiller import DistillerConfig


def _validate_positive_int(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or value <= 0):
        raise IntelligenceError(f"{name} must be greater than 0")


def _set_non_none(config: DistillerConfig, updates: dict[str, object]) -> None:
    for attr, value in updates.items():
        if value is not None:
            setattr(config, attr, value)


def apply_config_updates(  # pylint: disable=too-many-arguments
    config: DistillerConfig,
    *,
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
    max_concurrency: int | None = None,
    embedding_url: str | None = None,
    embedding_key: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    clear_embedding: bool = False,
) -> DistillerConfig:
    _validate_positive_int("timeout", timeout)
    _validate_positive_int("max_concurrency", max_concurrency)
    _set_non_none(
        config,
        {
            "api_base": url,
            "api_key": key,
            "model": model,
            "provider": provider,
            "timeout": timeout,
            "max_concurrency": max_concurrency,
        },
    )

    if clear_embedding:
        config.embedding_base = None
        config.embedding_key = None
        config.embedding_model = None
        config.embedding_provider = None

    _set_non_none(
        config,
        {
            "embedding_base": embedding_url,
            "embedding_key": embedding_key,
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
        },
    )

    # Re-run validators after raw assignment (blank → None)
    if isinstance(config, DistillerConfig):
        config = DistillerConfig.model_validate(config.model_dump())

    if (config.embedding_base is not None) != (config.embedding_model is not None):
        raise IntelligenceError(
            "embedding_base and embedding_model must both be set or both unset"
        )
    return config


def _has_updates(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    url,
    key,
    model,
    provider,
    timeout,
    max_concurrency,
    embedding_url,
    embedding_key,
    embedding_model,
    embedding_provider,
    clear_embedding,
) -> bool:
    return embedding_fields_requested(
        embedding_url=embedding_url,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        clear_embedding=clear_embedding,
    ) or distill_fields_requested(
        url=url,
        key=key,
        model=model,
        provider=provider,
        timeout=timeout,
        max_concurrency=max_concurrency,
    )


def embedding_fields_requested(  # pylint: disable=too-many-arguments
    *,
    embedding_url: str | None = None,
    embedding_key: str | None = None,
    embedding_model: str | None = None,
    embedding_provider: str | None = None,
    clear_embedding: bool = False,
) -> bool:
    """True when this config call touches remote/local embedder selection."""
    return any(
        [
            embedding_url is not None,
            embedding_key is not None,
            embedding_model is not None,
            embedding_provider is not None,
            clear_embedding,
        ]
    )


def distill_fields_requested(
    *,
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    **extra: object,
) -> bool:
    """True when this config call touches LLM distillation settings."""
    return any(
        val is not None
        for val in (
            url,
            key,
            model,
            provider,
            extra.get("timeout"),
            extra.get("max_concurrency"),
        )
    )


def load_or_update_config(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    url: str | None = None,
    key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    timeout: int | None = None,
    max_concurrency: int | None = None,
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
        timeout,
        max_concurrency,
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
        timeout=timeout,
        max_concurrency=max_concurrency,
        embedding_url=embedding_url,
        embedding_key=embedding_key,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        clear_embedding=clear_embedding,
    )
    config.save()
    return config
