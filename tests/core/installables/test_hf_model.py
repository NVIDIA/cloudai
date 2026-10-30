# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from unittest.mock import patch

import pytest

from cloudai.core import BaseInstaller, HFModel


@pytest.fixture
def hf_model() -> HFModel:
    return HFModel(model_name="some_model_name")


@pytest.fixture
def installer(slurm_system, tmp_path: Path) -> BaseInstaller:
    slurm_system.hf_home_path = tmp_path
    return BaseInstaller(slurm_system)


def test_download(hf_model: HFModel, installer: BaseInstaller) -> None:
    assert hf_model._installed_path is None

    with patch("huggingface_hub.snapshot_download", return_value=str("/real/path")):
        hf_model.install(installer)

    assert hf_model.installed_path == Path("/real/path")


def test_download_raises(hf_model: HFModel, installer: BaseInstaller) -> None:
    with patch("huggingface_hub.snapshot_download", side_effect=Exception("some error message")):
        result = hf_model.install(installer)

    assert not result.success
    assert "some error message" in result.message
    assert hf_model._installed_path is None


def test_is_downloaded(hf_model: HFModel, installer: BaseInstaller) -> None:
    with patch("huggingface_hub.snapshot_download", return_value=str("/real/path")):
        result = hf_model.is_installed(installer)

    assert result.success
    assert result.message == "/real/path"
    assert hf_model.installed_path == Path("/real/path")


def test_is_downloaded_raises(hf_model: HFModel, installer: BaseInstaller) -> None:
    with patch("huggingface_hub.snapshot_download", side_effect=Exception("some error message")):
        result = hf_model.is_installed(installer)

    assert not result.success
    assert "some error message" in result.message
    assert hf_model._installed_path is None


@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_revision_used_for_download_and_local_check(installer: BaseInstaller, revision: str | None):
    model = HFModel(model_name="example/model", revision=revision)
    with patch("huggingface_hub.snapshot_download", return_value="/snapshot") as download:
        assert model.install(installer).success
        assert model.is_installed(installer).success
    for call in download.call_args_list:
        assert call.kwargs["revision"] == revision
    assert download.call_args_list[1].kwargs["local_files_only"] is True


def test_revisions_are_distinct_installables():
    assert len({HFModel("model"), HFModel("model", revision="a"), HFModel("model", revision="b")}) == 3
    assert HFModel("model", revision="a") == HFModel("model", revision="a")


def make_cached_model(cache: Path, name: str) -> Path:
    repo = cache / f"models--example--{name}"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "shared").write_text("shared weights")
    (repo / "refs").mkdir()
    for revision, ref in [("a" * 40, "main"), ("b" * 40, "other")]:
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "weights").symlink_to("../../blobs/shared")
        (repo / "refs" / ref).write_text(revision)
    return repo


@pytest.mark.parametrize("revision", ["a" * 40, "main"])
def test_uninstall_revision_preserves_other_revisions_and_repositories(installer: BaseInstaller, revision: str):
    cache = installer.system.hf_home_path / "hub"
    repo = make_cached_model(cache, "model")
    other_repo = make_cached_model(cache, "other")
    if revision == "a" * 40:
        (repo / "refs" / "main").unlink()
    result = HFModel("example/model", revision=revision).uninstall(installer)
    assert result.success, result.message
    assert not (repo / "snapshots" / ("a" * 40)).exists()
    assert (repo / "snapshots" / ("b" * 40) / "weights").read_text() == "shared weights"
    assert not (repo / "refs" / "main").exists()
    assert (other_repo / "snapshots" / ("a" * 40) / "weights").exists()


def test_uninstall_missing_revision_preserves_model(installer: BaseInstaller):
    repo = make_cached_model(installer.system.hf_home_path / "hub", "model")
    assert HFModel("example/model", revision="missing").uninstall(installer).success
    assert (repo / "snapshots" / ("a" * 40) / "weights").exists()


def test_uninstall_revision_failure_does_not_delete_model(installer: BaseInstaller):
    make_cached_model(installer.system.hf_home_path / "hub", "model")
    with patch("huggingface_hub.scan_cache_dir", side_effect=OSError("unreadable")), patch("subprocess.run") as run:
        result = HFModel("example/model", revision="main").uninstall(installer)
    assert not result.success
    assert "unreadable" in result.message
    run.assert_not_called()


def test_uninstall_unpinned_keeps_whole_model_removal(installer: BaseInstaller):
    with patch("huggingface_hub.snapshot_download", return_value="/snapshot"), patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert HFModel("example/model").uninstall(installer).success
    assert run.call_args.args[0] == ["hf", "cache", "rm", "-y", "model/example/model"]


@pytest.mark.parametrize("revision", ["main", "stable", "a" * 40])
def test_uninstall_preserves_snapshot_with_other_refs(installer: BaseInstaller, revision: str):
    repo = make_cached_model(installer.system.hf_home_path / "hub", "model")
    (repo / "refs" / "stable").write_text("a" * 40)

    result = HFModel("example/model", revision=revision).uninstall(installer)

    assert not result.success
    assert "also referenced by" in result.message
    for ref in {"main", "stable"} - {revision}:
        assert ref in result.message
    for ref in ("main", "stable"):
        assert (repo / "refs" / ref).read_text() == "a" * 40
    assert (repo / "snapshots" / ("a" * 40) / "weights").read_text() == "shared weights"
    assert (repo / "snapshots" / ("b" * 40) / "weights").read_text() == "shared weights"
