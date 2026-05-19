import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from code_rag.discovery.providers.rust import RustDiscoveryProvider
from code_rag.core.models import KnowledgeUnit, UnitKind


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[])
    return parser


@pytest.fixture
def provider(mock_parser):
    return RustDiscoveryProvider(parser=mock_parser)


class TestRustDiscoveryDetailed:
    """Detailed tests for RustDiscoveryProvider to increase coverage."""

    def test_find_cargo(self, provider):
        """Test finding cargo binary."""
        with patch("shutil.which", return_value="/usr/bin/cargo"):
            assert provider._find_cargo() == "/usr/bin/cargo"

    @pytest.mark.asyncio
    async def test_find_crate_root_success(self, provider, tmp_path):
        """Test finding crate root using cargo metadata mock."""
        (tmp_path / "Cargo.toml").touch()

        mock_metadata = {
            "packages": [
                {"name": "test-crate", "manifest_path": str(tmp_path / "Cargo.toml")}
            ]
        }

        with patch(
            "code_rag.discovery.providers.rust.find_directory_upwards",
            return_value=tmp_path / "Cargo.toml",
        ), patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                json.dumps(mock_metadata).encode(),
                b"",
            )
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            root = await provider._find_crate_root("test-crate")
            assert root == tmp_path

    @pytest.mark.asyncio
    async def test_recursive_extract_following_modules(
        self, provider, mock_parser, tmp_path
    ):
        """Test recursive extraction follows 'mod' declarations."""
        lib_rs = tmp_path / "lib.rs"
        mod_rs = tmp_path / "my_mod.rs"
        lib_rs.touch()
        mod_rs.touch()

        # Unit for lib.rs: declares a module
        unit_mod = KnowledgeUnit(
            id="lib.rs:my_mod",
            name="my_mod",
            kind=UnitKind.MODULE,
            path=str(lib_rs),
            code_hash="h1",
            metadata={"node_type": "mod_item", "raw_code": "pub mod my_mod;"},
        )

        # Unit for my_mod.rs: contains a public function
        unit_fn = KnowledgeUnit(
            id="my_mod.rs:api_call",
            name="api_call",
            kind=UnitKind.FUNCTION,
            path=str(mod_rs),
            code_hash="h2",
            signature="pub fn api_call()",
        )

        async def mock_distill(path):
            if "lib.rs" in path:
                return [unit_mod]
            if "my_mod.rs" in path:
                return [unit_fn]
            return []

        mock_parser.distill_file.side_effect = mock_distill

        visited = set()
        results = []
        await provider._recursive_extract(lib_rs, visited, results, depth=0)

        # Results should contain api_call from my_mod.rs
        assert len(results) == 1
        assert results[0].name == "api_call"

    def test_resolve_mod_path_different_styles(self, provider, tmp_path):
        """Test resolving Rust module paths (file vs directory style)."""
        # Style 1: name.rs
        (tmp_path / "foo.rs").touch()
        assert provider._resolve_mod_path(tmp_path, "foo") == tmp_path / "foo.rs"

        # Style 2: name/mod.rs
        bar_dir = tmp_path / "bar"
        bar_dir.mkdir()
        (bar_dir / "mod.rs").touch()
        assert provider._resolve_mod_path(tmp_path, "bar") == bar_dir / "mod.rs"

    @pytest.mark.asyncio
    async def test_extract_api_integration_mocked(self, provider, tmp_path):
        """High-level test of extract_api with mocked sub-calls."""
        crate_root = tmp_path / "crate"
        crate_root.mkdir()
        src_dir = crate_root / "src"
        src_dir.mkdir()
        (src_dir / "lib.rs").touch()

        with patch.object(
            provider, "_find_crate_root", return_value=crate_root
        ), patch.object(
            provider, "_recursive_extract", side_effect=AsyncMock()
        ) as mock_rec:

            async def fake_rec(path, visited, results, depth):
                results.append(
                    KnowledgeUnit(
                        id="id",
                        name="MyStruct",
                        kind=UnitKind.CLASS,
                        path=str(path),
                        code_hash="h",
                        signature="pub struct MyStruct",
                        docstring="Documentation",
                    )
                )

            mock_rec.side_effect = fake_rec

            report = await provider.extract_api("test-crate")
            assert "Rust Crate 'test-crate'" in report
            assert "MyStruct" in report
            assert "Documentation" in report
