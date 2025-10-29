from unittest import mock

from stck import updater


def test_choose_strategy_prefers_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    assert updater.choose_strategy(True, False) == "git"
    assert updater.choose_strategy(False, True) == "pip"


def test_choose_strategy_detects_git(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    assert updater.choose_strategy(False, False) == "git"


def test_choose_strategy_defaults_to_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, "PROJECT_ROOT", tmp_path)
    assert updater.choose_strategy(False, False) == "pip"


def test_update_with_git_invokes_subprocess(monkeypatch):
    with mock.patch.object(updater, "_run_command", return_value=0) as run_mock:
        result = updater.update_with_git()
    assert result == 0
    run_mock.assert_called_once()


def test_update_with_pip_invokes_subprocess(monkeypatch):
    with mock.patch.object(updater, "_run_command", return_value=0) as run_mock:
        result = updater.update_with_pip()
    assert result == 0
    run_mock.assert_called_once()
