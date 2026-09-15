# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib.util
import json
import shlex
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / ".agents/skills/cloudai-remote-install/scripts/deploy.py"
SPEC = importlib.util.spec_from_file_location("cloudai_remote_deploy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("*.pyc\n.venv/\n.ruff_cache/\nresults/\n")
    (root / "src/cloudai").mkdir(parents=True)
    (root / "src/cloudai/main.py").write_text("print('original')\n")
    (root / "pyproject.toml").write_text('[project]\nname = "cloudai"\n')
    (root / "uv.lock").write_text("version = 1\n")
    deploy.git(root, "add", ".")
    deploy.git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "Initial",
    )
    return root


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "remote-home"
    home.mkdir()
    config = home / "system.toml"
    config.write_text('name = "test"\n')
    runner = Mock()
    monkeypatch.setattr(deploy.subprocess, "run", runner)
    return home, config, runner


def stage_snapshot(stage: Path, files: dict[str, str], *, main: bool = False, identity: str = "a" * 16) -> dict:
    stage.mkdir(exist_ok=True)
    record = {"checkout_id": identity, "main": main, "revision": "test", "files": {}}
    for name, contents in files.items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        record["files"][name] = deploy.signature(path)
    deploy.write_json(stage / deploy.STATE, record)
    return record


def test_checkout_identity_and_selective_snapshot(checkout: Path, tmp_path: Path):
    main_record, _ = deploy.snapshot(checkout, [])
    worktree = tmp_path / "linked"
    deploy.git(checkout, "worktree", "add", "--detach", str(worktree))
    (worktree / "src/cloudai/main.py").write_text("print('local changes')\n")
    (worktree / "src/cloudai/new.py").write_text("# untracked task code\n")
    (worktree / "src/cloudai/cache.pyc").write_bytes(b"bytecode")
    (worktree / "results/task").mkdir(parents=True)
    (worktree / "results/task/input.txt").write_text("task data")
    (worktree / "results/task/__pycache__").mkdir()
    (worktree / "results/task/__pycache__/cache.pyc").touch()
    record, files = deploy.snapshot(worktree, ["results/task"])
    assert main_record["main"] and not record["main"]
    assert main_record["checkout_id"] != record["checkout_id"]
    assert set(files) == {
        "src/cloudai/main.py",
        "src/cloudai/new.py",
        "pyproject.toml",
        "uv.lock",
        "results/task/input.txt",
        deploy.HELPER,
    }
    assert files["src/cloudai/main.py"].read_text() == "print('local changes')\n"
    deploy.git(worktree, "switch", "-c", "another-branch")
    assert deploy.snapshot(worktree, [])[0]["checkout_id"] == record["checkout_id"]
    assert deploy.destination(tmp_path, main_record) == tmp_path / "cloudai"
    assert deploy.destination(tmp_path, record) == tmp_path / "cloudai-worktrees" / record["checkout_id"]


@pytest.mark.parametrize("include", ["../outside", "/tmp", "src/cloudai/linked"])
def test_reject_escaping_includes(checkout: Path, tmp_path: Path, include: str):
    (checkout / "src/cloudai/linked").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        deploy.selected_files(checkout, [include])


def test_apply_update_preserves_user_files_config_and_environment(
    remote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home, config, runner = remote
    stage = tmp_path / "stage"
    stage_snapshot(stage, {"src/main.py": "old", "src/obsolete.py": "old"})
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/some-other-checkout/.venv")
    monkeypatch.setenv("VIRTUAL_ENV", "/another/environment")
    target = deploy.apply_snapshot(stage, home, str(config), ".bashrc", ["dev"])
    (target / "notes.txt").write_text("user notes")
    (target / deploy.KEEP).touch()
    stage_snapshot(stage, {"src/main.py": "updated", "src/new.py": "new"})
    deploy.apply_snapshot(stage, home, str(config), ".bashrc", ["dev"])
    assert (target / "src/main.py").read_text() == "updated"
    assert not (target / "src/obsolete.py").exists()
    assert (target / "notes.txt").read_text() == "user notes"
    assert (target / deploy.KEEP).exists()
    assert config.read_text() == 'name = "test"\n'
    assert (home / ".bashrc").read_text().count("export CLOUDAI_SYSTEM_CONFIG=") == 1
    record = json.loads((target / deploy.STATE).read_text())
    assert record["ready"]
    sync, verify = runner.call_args_list[-2:]
    assert sync.args[0] == ["uv", "sync", "--locked", "--extra", "dev"]
    assert verify.args[0] == ["uv", "run", "--locked", "--no-sync", "cloudai", "verify-configs", str(config)]
    assert sync.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(target / ".venv")
    assert "VIRTUAL_ENV" not in sync.kwargs["env"]
    assert sync.kwargs["cwd"] == target


@pytest.mark.parametrize("conflict", ["edit", "deleted", "unmanaged", "symlink", "venv"])
def test_remote_conflicts_abort_before_writing(remote, tmp_path: Path, conflict: str):
    home, config, runner = remote
    stage = tmp_path / "stage"
    stage_snapshot(stage, {"src/main.py": "original"})
    target = deploy.apply_snapshot(stage, home, str(config), None, [])
    if conflict == "edit":
        (target / "src/main.py").write_text("remote edit")
    elif conflict == "deleted":
        (target / "src/main.py").unlink()
    elif conflict == "unmanaged":
        (target / "new.py").write_text("user file")
    elif conflict == "symlink":
        (target / "new.py").symlink_to(config)
    else:
        (target / ".venv").symlink_to(home)
    before = (target / deploy.STATE).read_text()
    stage_snapshot(stage, {"src/main.py": "new", "new.py": "incoming"})
    runner.reset_mock()
    with pytest.raises(ValueError):
        deploy.apply_snapshot(stage, home, str(config), None, [])
    runner.assert_not_called()
    assert (target / deploy.STATE).read_text() == before
    assert config.read_text() == 'name = "test"\n'


def test_main_ownership_and_unmanaged_clone(remote, tmp_path: Path):
    home, config, _ = remote
    stage = tmp_path / "stage"
    (home / "cloudai").mkdir()
    (home / "cloudai/.git").mkdir()
    stage_snapshot(stage, {"main.py": "source"}, main=True)
    with pytest.raises(ValueError, match="Unmanaged directory"):
        deploy.apply_snapshot(stage, home, str(config), None, [])
    (home / "cloudai/.git").rmdir()
    target = deploy.apply_snapshot(stage, home, str(config), None, [])
    stage_snapshot(stage, {"main.py": "different"}, main=True, identity="b" * 16)
    with pytest.raises(ValueError, match="Different checkout"):
        deploy.apply_snapshot(stage, home, str(config), None, [])
    assert (target / "main.py").read_text() == "source"


def test_failed_uv_install_can_be_retried_without_marking_ready(remote, tmp_path: Path):
    home, config, runner = remote
    stage = tmp_path / "stage"
    stage_snapshot(stage, {"main.py": "source"})
    runner.side_effect = subprocess.CalledProcessError(1, ["uv", "sync"])
    with pytest.raises(subprocess.CalledProcessError):
        deploy.apply_snapshot(stage, home, str(config), ".bashrc", [])
    target = home / "cloudai-worktrees" / ("a" * 16)
    assert not json.loads((target / deploy.STATE).read_text())["ready"]
    assert not (home / ".bashrc").exists()
    runner.side_effect = None
    deploy.apply_snapshot(stage, home, str(config), ".bashrc", [])
    assert json.loads((target / deploy.STATE).read_text())["ready"]


def test_default_export_and_explicit_feature_override(remote):
    home, config, _ = remote
    rc = home / ".zshrc"
    rc.write_text("# existing shell setup\n")
    path, content = deploy.shell_export(home, config, ".zshrc")
    path.write_text(content)
    assert deploy.shell_export(home, config, ".zshrc")[1] == content
    alternative = home / "cloudai-worktrees" / ("a" * 16) / "system.toml"
    alternative.parent.mkdir(parents=True)
    alternative.write_text("# feature config")
    env, export = deploy.configuration(home, str(alternative), None)
    assert env["CLOUDAI_SYSTEM_CONFIG"] == str(alternative) and export is None
    with pytest.raises(ValueError, match="outside deployment"):
        deploy.configuration(home, str(alternative), ".zshrc")
    assert rc.read_text() == content
    rc.write_text("export CLOUDAI_SYSTEM_CONFIG=/existing/system.toml\n")
    with pytest.raises(ValueError, match="already sets"):
        deploy.shell_export(home, config, ".zshrc")


def test_cleanup_age_main_protection_and_observed_use(remote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home, config, _ = remote
    stage = tmp_path / "stage"
    stage_snapshot(stage, {"main.py": "source"})
    worktree = deploy.apply_snapshot(stage, home, str(config), None, [])
    stage_snapshot(stage, {"main.py": "main source"}, main=True)
    main = deploy.apply_snapshot(stage, home, str(config), None, [])
    monkeypatch.setattr(deploy.time, "time", Mock(return_value=time.time() + 22 * 86400))
    rows = deploy.maintenance(home, "status", None)
    assert [row["review_cleanup"] for row in rows] == [False, True]
    assert rows[0]["protected"] and rows[0]["directory"] == str(main)
    deploy.maintenance(home, "mark-used", worktree)
    assert not deploy.maintenance(home, "status", None)[1]["review_cleanup"]
    deploy.maintenance(home, "protect", worktree)
    assert (worktree / deploy.KEEP).exists()
    assert deploy.maintenance(home, "status", None)[1]["protected"]
    record = json.loads((worktree / deploy.STATE).read_text())
    del record["last_used"]
    deploy.write_json(worktree / deploy.STATE, record)
    assert deploy.maintenance(home, "status", None)[1]["days_since_observed_use"] is None


def test_tampered_snapshot_and_symlinked_worktree_root(remote, tmp_path: Path):
    home, config, runner = remote
    stage = tmp_path / "stage"
    record = stage_snapshot(stage, {"main.py": "source"})
    (stage / "main.py").write_text("changed")
    with pytest.raises(ValueError, match="Snapshot changed"):
        deploy.apply_snapshot(stage, home, str(config), None, [])
    runner.assert_not_called()
    (home / "cloudai-worktrees").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        deploy.destination(home, record)


def test_same_checkout_deployments_are_serialized(remote, tmp_path: Path):
    home, config, runner = remote
    stage = tmp_path / "stage"
    record = stage_snapshot(stage, {"main.py": "source"})
    target = deploy.destination(home, record)
    target.mkdir(parents=True)
    with deploy.deployment_lock(target), pytest.raises(BlockingIOError):
        deploy.apply_snapshot(stage, home, str(config), None, [])
    runner.assert_not_called()
    assert not (target / "main.py").exists()


def test_deploy_transport_selects_payload_and_quotes_remote_arguments(checkout: Path, monkeypatch: pytest.MonkeyPatch):
    read_command = deploy.subprocess.check_output
    run_command = deploy.subprocess.run
    remote_stage = "/home/test/.cloudai-stage-abcdefgh"
    calls = []

    def output(command, **kwargs):
        if command[0] == "ssh":
            assert command == ["ssh", "test-cluster", 'mktemp -d "$HOME/.cloudai-stage-XXXXXXXX"']
            return remote_stage + "\n"
        return read_command(command, **kwargs)

    def run(command, **kwargs):
        if command[0] == "git":
            return run_command(command, **kwargs)
        calls.append(command)
        if command[0] == "rsync":
            stage = Path(command[-2])
            record = json.loads((stage / deploy.STATE).read_text())
            assert set(record["files"]) == {"pyproject.toml", "uv.lock", "src/cloudai/main.py", deploy.HELPER}
            assert all(deploy.signature(stage / name) == value for name, value in record["files"].items())
            assert not (stage / ".git").exists()

    monkeypatch.setattr(deploy.subprocess, "check_output", output)
    monkeypatch.setattr(deploy.subprocess, "run", run)
    args = argparse.Namespace(
        host="test-cluster",
        checkout=checkout,
        include=[],
        system_config="/home/test/config's space/system.toml",
        shell_rc=".zshrc",
        extra=["dev"],
    )
    deploy.deploy(args)
    assert calls[0][:3] == ["rsync", "-a", "--"]
    assert calls[0][-1] == f"test-cluster:{remote_stage}/"
    assert calls[1][:2] == ["ssh", "test-cluster"]
    remote_command = shlex.split(calls[1][2])
    assert remote_command[1:8] == [
        "uv",
        "run",
        "--no-project",
        "--python",
        ">=3.10",
        "python",
        f"{remote_stage}/{deploy.HELPER}",
    ]
    assert remote_command[-6:] == ["--system-config", args.system_config, "--shell-rc", ".zshrc", "--extra", "dev"]


def test_shell_edits_during_install_are_preserved(remote, tmp_path: Path):
    home, config, runner = remote
    stage = tmp_path / "stage"
    stage_snapshot(stage, {"main.py": "source"})
    runner.side_effect = lambda *args, **kwargs: (home / ".bashrc").write_text("# edited during install\n")
    with pytest.raises(ValueError, match="changed during installation"):
        deploy.apply_snapshot(stage, home, str(config), ".bashrc", [])
    assert (home / ".bashrc").read_text() == "# edited during install\n"
