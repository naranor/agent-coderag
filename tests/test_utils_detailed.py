import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from code_rag.core.utils import validate_path, find_directory_upwards


class TestUtilsDetailed:
    """Detailed tests for utility functions with robust mocking."""

    def test_validate_path_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "test.txt"
            path.touch()
            # Should not raise
            assert validate_path(path) == path.resolve()

    def test_validate_path_traversal_attempt(self, tmp_path):
        # We'll use a real directory and an explicitly outside directory
        # to trigger the ValueError without complex mocks

        root = tmp_path / "project"
        root.mkdir()

        outside = tmp_path / "unauthorized"
        outside.mkdir()
        outside_file = outside / "steal.txt"
        outside_file.touch()

        # We must mock home and tempdir to ensure they don't match our outside path
        with patch(
            "tempfile.gettempdir", return_value=str(tmp_path / "other_temp")
        ), patch("pathlib.Path.home", return_value=tmp_path / "other_home"), patch(
            "code_rag.core.utils.get_global_dir"
        ) as mock_global:
            mock_global.return_value = tmp_path / "other_global"

            with pytest.raises(ValueError) as exc:
                # pass root explicitly to avoid cwd/git detection
                validate_path(outside_file, root=root)
            assert "Security Risk" in str(exc.value)

    def test_validate_path_nonexistent(self, tmp_path):
        # Even if it doesn't exist, we check the parent
        root = tmp_path / "safe"
        root.mkdir()

        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "new_file.db"

        with patch(
            "tempfile.gettempdir", return_value=str(tmp_path / "other_temp")
        ), patch("pathlib.Path.home", return_value=tmp_path / "other_home"), patch(
            "code_rag.core.utils.get_global_dir"
        ) as mock_global:
            mock_global.return_value = tmp_path / "other_global"

            with pytest.raises(ValueError):
                validate_path(outside_file, root=root)

    def test_find_directory_upwards_success(self, tmp_path):
        # Create structure: /root/sub/sub2
        # Target: /root/target.txt
        root = tmp_path / "root"
        root.mkdir()
        target = root / "find_me"
        target.touch()

        sub = root / "sub" / "sub2"
        sub.mkdir(parents=True)

        result = find_directory_upwards(sub, "find_me")
        assert result == target

    def test_find_directory_upwards_failure(self, tmp_path):
        result = find_directory_upwards(tmp_path, "missing_file_xyz")
        assert result is None
