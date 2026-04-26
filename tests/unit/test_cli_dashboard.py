"""Unit tests for the `askbook dashboard` CLI subcommand."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from askbook.cli import app

runner = CliRunner()


def test_dashboard_command_invokes_streamlit_run() -> None:
    """dashboard command must call subprocess.run with streamlit run <app.py>."""
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        result = runner.invoke(app, ["dashboard"])

    assert result.exit_code == 0, result.output
    assert captured, "subprocess.run was never called"
    cmd = captured[0]
    assert "streamlit" in cmd
    assert "run" in cmd
    # app.py must be somewhere in the command
    assert any("app.py" in arg for arg in cmd)


def test_dashboard_command_accepts_port_option() -> None:
    """dashboard --port 9000 must pass --server.port 9000 to streamlit."""
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0)

    with patch("subprocess.run", side_effect=fake_run):
        result = runner.invoke(app, ["dashboard", "--port", "9000"])

    assert result.exit_code == 0, result.output
    assert captured
    cmd = captured[0]
    assert "--server.port" in cmd
    assert "9000" in cmd
