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
from unittest.mock import MagicMock, patch

import pytest

from cloudai.util.object_store import ObjectStore, S3ObjectStore, UploadStats, join_key


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "scenario_2025-04-16_14-27-45"
    (root / "test-a" / "0").mkdir(parents=True)
    (root / "test-a" / "0" / "stdout.txt").write_text("out")
    (root / "test-a" / "0" / "stderr.txt").write_text("err!!")
    (root / "report.html").write_text("<html></html>")
    return root


class RecordingStore(ObjectStore):
    """In-memory ObjectStore that records uploads, to exercise upload_directory()."""

    def __init__(self, fail_on: str = "") -> None:
        self.uploads: list[tuple[Path, str]] = []
        self.fail_on = fail_on

    def uri(self, key: str) -> str:
        return f"mem://{key}"

    def upload_file(self, local_path: Path, key: str) -> None:
        if self.fail_on and self.fail_on in key:
            raise RuntimeError("boom")
        self.uploads.append((local_path, key))


@pytest.mark.parametrize(
    "parts,expected",
    [
        (("a", "b"), "a/b"),
        (("", "b"), "b"),
        (("a/", "/b"), "a/b"),
        (("", ""), ""),
        (("bucket", "", "key"), "bucket/key"),
    ],
)
def test_join_key(parts: tuple[str, ...], expected: str) -> None:
    assert join_key(*parts) == expected


def test_upload_directory_preserves_relative_paths(tree: Path) -> None:
    store = RecordingStore()
    stats = store.upload_directory(tree, "runs")

    assert sorted(key for _, key in store.uploads) == [
        "runs/report.html",
        "runs/test-a/0/stderr.txt",
        "runs/test-a/0/stdout.txt",
    ]
    assert stats.files_uploaded == 3
    assert stats.bytes_uploaded == len("<html></html>") + len("err!!") + len("out")
    assert stats.is_successful


def test_upload_directory_without_prefix(tree: Path) -> None:
    store = RecordingStore()
    store.upload_directory(tree)

    assert "report.html" in [key for _, key in store.uploads]


def test_upload_directory_honours_exclude(tree: Path) -> None:
    store = RecordingStore()
    stats = store.upload_directory(tree, "runs", exclude=["*.txt"])

    assert [key for _, key in store.uploads] == ["runs/report.html"]
    assert stats.files_uploaded == 1


def test_upload_directory_collects_failures_and_continues(tree: Path) -> None:
    store = RecordingStore(fail_on="stdout.txt")
    stats = store.upload_directory(tree, "runs")

    assert stats.files_uploaded == 2
    assert not stats.is_successful
    assert len(stats.failures) == 1
    failed_path, message = stats.failures[0]
    assert failed_path.name == "stdout.txt"
    assert "boom" in message


def test_upload_directory_skips_empty_dirs(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    (root / "nested").mkdir(parents=True)

    stats = RecordingStore().upload_directory(root, "runs")

    assert stats == UploadStats()


def test_s3_object_store_uri() -> None:
    store = S3ObjectStore(bucket="my-bucket")
    assert store.uri("runs/report.html") == "s3://my-bucket/runs/report.html"


def test_s3_object_store_upload_file_and_client_reuse(tmp_path: Path) -> None:
    local = tmp_path / "report.html"
    local.write_text("<html></html>")

    with patch("cloudai.util.object_store.lazy") as mock_lazy:
        client = MagicMock()
        mock_lazy.boto3.client.return_value = client

        store = S3ObjectStore(bucket="my-bucket", endpoint_url="http://localhost:9000", region="us-east-1")
        store.upload_file(local, "runs/report.html")
        store.upload_file(local, "runs/again.html")

        mock_lazy.boto3.client.assert_called_once_with(
            "s3", endpoint_url="http://localhost:9000", region_name="us-east-1"
        )
        client.upload_file.assert_any_call(str(local), "my-bucket", "runs/report.html")
        assert client.upload_file.call_count == 2


def test_s3_object_store_exists() -> None:
    with patch("cloudai.util.object_store.lazy") as mock_lazy:
        client = MagicMock()
        mock_lazy.boto3.client.return_value = client

        store = S3ObjectStore(bucket="my-bucket")
        assert store.exists("present") is True

        client.head_object.side_effect = RuntimeError("404")
        assert store.exists("missing") is False
