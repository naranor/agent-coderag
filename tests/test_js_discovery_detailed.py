import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from code_rag.discovery.providers.javascript import JavaScriptDiscoveryProvider
from code_rag.core.models import KnowledgeUnit, UnitKind
from code_rag.core.exceptions import DiscoveryError


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.distill_file = AsyncMock(return_value=[])
    return parser


@pytest.fixture
def provider(mock_parser):
    return JavaScriptDiscoveryProvider(parser=mock_parser)


class TestJSDiscoveryDetailed:
    """Detailed tests for JavaScriptDiscoveryProvider to increase coverage."""

    @pytest.mark.asyncio
    async def test_find_package_root(self, provider, tmp_path):
        """Test finding package root in node_modules."""
        nm_dir = tmp_path / "node_modules"
        nm_dir.mkdir()
        lib_dir = nm_dir / "test-lib"
        lib_dir.mkdir()

        with patch(
            "code_rag.discovery.providers.javascript.find_directory_upwards",
            return_value=nm_dir,
        ):
            root = provider._find_package_root("test-lib")
            assert root == lib_dir

    def test_get_target_files_from_package_json(self, provider, tmp_path):
        """Test identifying entry points from package.json."""
        pkg_json = tmp_path / "package.json"
        pkg_data = {"types": "dist/index.d.ts", "main": "dist/index.js"}
        pkg_json.write_text(json.dumps(pkg_data))

        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "index.d.ts").touch()
        (tmp_path / "dist" / "index.js").touch()

        targets = provider._get_target_files(tmp_path)
        # Current implementation stops if types found
        assert len(targets) == 1
        assert targets[0].name == "index.d.ts"

    def test_get_target_files_fallback_to_main(self, provider, tmp_path):
        """Test fallback to main if no types found."""
        pkg_json = tmp_path / "package.json"
        pkg_data = {"main": "index.js"}
        pkg_json.write_text(json.dumps(pkg_data))
        (tmp_path / "index.js").touch()

        targets = provider._get_target_files(tmp_path)
        assert len(targets) == 1
        assert targets[0].name == "index.js"

    @pytest.mark.asyncio
    async def test_recursive_extract_following_exports(
        self, provider, mock_parser, tmp_path
    ):
        """Test that recursive extract follows local exports."""
        file1 = tmp_path / "index.ts"
        file2 = tmp_path / "utils.ts"
        file1.touch()
        file2.touch()

        unit_export = KnowledgeUnit(
            id="index.ts:module",
            name="module",
            kind=UnitKind.MODULE,
            path=str(file1),
            code_hash="h1",
            metadata={
                "node_type": "export_statement",
                "raw_code": "export * from './utils'",
            },
        )
        unit_fn = KnowledgeUnit(
            id="utils.ts:helper",
            name="helper",
            kind=UnitKind.FUNCTION,
            path=str(file2),
            code_hash="h2",
            signature="export function helper()",
        )

        async def mock_distill(path):
            if "index.ts" in path:
                return [unit_export]
            if "utils.ts" in path:
                return [unit_fn]
            return []

        mock_parser.distill_file.side_effect = mock_distill

        visited = set()
        results = []
        await provider._recursive_extract(file1, visited, results, depth=0)

        assert len(results) == 1
        assert results[0].name == "helper"

    @pytest.mark.asyncio
    async def test_extract_api_definitely_typed_fallback(
        self, provider, mock_parser, tmp_path
    ):
        """Test fallback to @types package."""
        nm_dir = tmp_path / "node_modules"
        nm_dir.mkdir()
        types_dir = nm_dir / "@types" / "mylib"
        types_dir.mkdir(parents=True)
        index_d_ts = types_dir / "index.d.ts"
        index_d_ts.touch()

        # Ensure it finds something
        mock_parser.distill_file.return_value = [
            KnowledgeUnit(
                id="u1",
                name="func",
                kind=UnitKind.FUNCTION,
                path=str(index_d_ts),
                code_hash="h",
                signature="declare function func()",
            )
        ]

        with patch(
            "code_rag.discovery.providers.javascript.find_directory_upwards",
            return_value=nm_dir,
        ):
            report = await provider.extract_api("mylib")
            assert "JavaScript/TypeScript Library 'mylib'" in report
            assert "func" in report

    @pytest.mark.asyncio
    async def test_extract_api_discovery_error(self, provider, tmp_path):
        """Test error when package not found."""
        with patch(
            "code_rag.discovery.providers.javascript.find_directory_upwards",
            return_value=tmp_path,
        ):
            with pytest.raises(DiscoveryError):
                await provider.extract_api("nonexistent")
