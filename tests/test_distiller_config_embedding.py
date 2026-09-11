import json
import logging
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from code_rag.core.exceptions import IntelligenceError
from code_rag.intelligence.distiller import DistillerConfig
from code_rag.services.config import apply_config_updates, load_or_update_config


def test_embedding_fields_default_none():
    cfg = DistillerConfig()
    assert cfg.embedding_base is None
    assert cfg.embedding_key is None
    assert cfg.embedding_model is None
    assert cfg.embedding_provider is None


def test_embedding_fields_trim_and_blank_to_none():
    cfg = DistillerConfig(
        embedding_base="  http://x  ",
        embedding_key="  ",
        embedding_model="  m  ",
        embedding_provider="",
    )
    assert cfg.embedding_base == "http://x"
    assert cfg.embedding_key is None
    assert cfg.embedding_model == "m"
    assert cfg.embedding_provider is None


def test_model_dump_is_flat_additive():
    dump = DistillerConfig(
        embedding_base="http://e", embedding_model="emb"
    ).model_dump()
    assert "embedding" not in dump or not isinstance(dump.get("embedding"), dict)
    assert dump["embedding_base"] == "http://e"
    assert dump["embedding_model"] == "emb"
    assert dump["model"] == "auto"


def test_config_url_only_does_not_wipe_embedding_fields(tmp_path):
    global_dir = tmp_path / "agent-coderag"
    global_dir.mkdir()
    payload = {
        "model": "auto",
        "api_base": "http://old",
        "api_key": None,
        "provider": "openai",
        "temperature": 0.0,
        "embedding_base": "http://embed",
        "embedding_key": "ek",
        "embedding_model": "text-emb",
        "embedding_provider": "openai",
    }
    (global_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ):
        cfg = load_or_update_config(url="http://distiller-only")
        cfg.save()
        reloaded = DistillerConfig.load()
    assert reloaded.api_base == "http://distiller-only"
    assert reloaded.embedding_base == "http://embed"
    assert reloaded.embedding_model == "text-emb"
    assert reloaded.embedding_key == "ek"


def test_load_or_update_config_trims_embedding_before_save(tmp_path):
    global_dir = tmp_path / "agent-coderag"
    global_dir.mkdir()
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ):
        load_or_update_config(
            embedding_url="  http://e  ",
            embedding_model="  m  ",
        )
        saved = json.loads((global_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["embedding_base"] == "http://e"
    assert saved["embedding_model"] == "m"


def test_embedding_only_update_writes(tmp_path):
    global_dir = tmp_path / "agent-coderag"
    global_dir.mkdir()
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ):
        out = load_or_update_config(
            embedding_url="http://e", embedding_model="m1", embedding_key="k"
        )
        assert out.embedding_base == "http://e"
        assert out.embedding_model == "m1"
        saved = json.loads((global_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["embedding_base"] == "http://e"
    assert saved["embedding_model"] == "m1"


def test_clear_embedding_then_set_flags():
    cfg = DistillerConfig(
        embedding_base="http://old",
        embedding_model="old-m",
        embedding_key="old-k",
        embedding_provider="openai",
    )
    apply_config_updates(
        cfg,
        clear_embedding=True,
        embedding_url="http://new",
        embedding_model="new-m",
    )
    assert cfg.embedding_base == "http://new"
    assert cfg.embedding_model == "new-m"
    assert cfg.embedding_key is None
    assert cfg.embedding_provider is None


def test_clear_embedding_alone_resets_all_four():
    cfg = DistillerConfig(
        embedding_base="http://old",
        embedding_model="old-m",
        embedding_key="old-k",
        embedding_provider="openai",
    )
    apply_config_updates(cfg, clear_embedding=True)
    assert cfg.embedding_base is None
    assert cfg.embedding_model is None
    assert cfg.embedding_key is None
    assert cfg.embedding_provider is None


def test_partial_embedding_raises_and_does_not_save(tmp_path):
    global_dir = tmp_path / "agent-coderag"
    global_dir.mkdir()
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ):
        with pytest.raises(IntelligenceError, match="both be set or both unset"):
            load_or_update_config(embedding_url="http://only-base")
        assert not (global_dir / "config.json").exists()


def test_load_or_update_config_show_does_not_save():
    with patch("code_rag.services.config.DistillerConfig.load") as mock_load:
        cfg = DistillerConfig()
        mock_load.return_value = cfg
        with patch.object(cfg, "save") as mock_save:
            assert load_or_update_config() is cfg
            mock_save.assert_not_called()


def test_non_string_embedding_field_rejected():
    with pytest.raises(ValidationError):
        DistillerConfig(embedding_base=123)  # type: ignore[arg-type]


def test_load_invalid_json_falls_back_to_defaults(tmp_path):
    global_dir = tmp_path / "agent-coderag"
    global_dir.mkdir()
    (global_dir / "config.json").write_text("{not-json", encoding="utf-8")
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ):
        cfg = DistillerConfig.load()
    assert cfg.model == "auto"
    assert cfg.embedding_base is None


def test_save_xor_raises_without_writing(tmp_path):
    global_dir = tmp_path / "agent-coderag"
    cfg = DistillerConfig()
    cfg.embedding_base = "http://e"
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ):
        with pytest.raises(IntelligenceError, match="both be set or both unset"):
            cfg.save()
    assert not (global_dir / "config.json").exists()


def test_save_io_error_is_logged(tmp_path, caplog):
    global_dir = tmp_path / "agent-coderag"
    cfg = DistillerConfig()
    with patch(
        "code_rag.intelligence.distiller.get_global_dir", return_value=global_dir
    ), patch("builtins.open", side_effect=OSError("disk full")):
        with caplog.at_level(logging.ERROR, logger="code_rag.intelligence.distiller"):
            cfg.save()
    assert any("Failed to save config" in rec.message for rec in caplog.records)
