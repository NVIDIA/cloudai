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

import json
import os
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from cloudai import TestScenario
from cloudai.core import PreparedS3Upload, Registry, S3UploadRequest
from cloudai.handlers import generate_reports
from cloudai.s3_reporter import S3UploadConfig, S3UploadReporter
from cloudai.systems.slurm.slurm_system import SlurmSystem
from cloudai.util.object_store import UploadStats


class TestS3UploadReporter:
    """Tests for uploading a results directory to object storage."""

    @pytest.fixture
    def results_dir(self, tmp_path: Path) -> Path:
        results_dir = tmp_path / "nccl-test_2025-04-16_14-27-45"
        (results_dir / "nccl" / "0").mkdir(parents=True)
        (results_dir / "nccl" / "0" / "stdout.txt").write_text("out")
        (results_dir / "report.html").write_text("<html></html>")
        (results_dir / "experiment.json").write_text(
            json.dumps(
                {
                    "id": results_dir.name,
                    "name": "dummy",
                    "system_name": "test_system",
                    "path": str(results_dir),
                    "user": "test-user",
                }
            )
        )
        return results_dir

    def reporter(self, slurm_system: SlurmSystem, results_dir: Path, **kwargs: Any) -> S3UploadReporter:
        return S3UploadReporter(
            slurm_system,
            TestScenario(name="dummy", test_runs=[]),
            results_dir,
            S3UploadConfig(enable=True, **kwargs),
        )

    @pytest.mark.parametrize("enabled", [True, False])
    def test_uploads_tree(self, slurm_system: SlurmSystem, results_dir: Path, enabled: bool) -> None:
        with (
            patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls,
            patch("getpass.getuser", return_value="another-user"),
        ):
            store = mock_store_cls.return_value
            store.upload_directory.return_value = UploadStats(files_uploaded=2, bytes_uploaded=16)

            reporter = self.reporter(slurm_system, results_dir, bucket="my-bucket", prefix="cloudai")
            reporter.config.enable = enabled
            reporter.generate()

            mock_store_cls.assert_called_once_with(bucket="my-bucket", endpoint_url=None, region=None)
            store.upload_directory.assert_called_once_with(
                results_dir, "cloudai/test_system/test-user/nccl-test_2025-04-16_14-27-45", max_workers=8
            )

    @pytest.mark.parametrize(
        "owner_state", ["missing-file", "invalid-json", "invalid-model", "missing-user", "empty-user"]
    )
    def test_missing_owner_falls_back_to_current_user(
        self, slurm_system: SlurmSystem, results_dir: Path, owner_state: str
    ) -> None:
        experiment_path = results_dir / "experiment.json"
        if owner_state == "missing-file":
            experiment_path.unlink()
        elif owner_state == "invalid-json":
            experiment_path.write_text("not JSON")
        elif owner_state == "invalid-model":
            experiment_path.write_text("{}")
        else:
            experiment = json.loads(experiment_path.read_text())
            if owner_state == "missing-user":
                del experiment["user"]
            else:
                experiment["user"] = ""
            experiment_path.write_text(json.dumps(experiment))

        with (
            patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls,
            patch("getpass.getuser", return_value="current-user"),
        ):
            store = mock_store_cls.return_value
            store.upload_directory.return_value = UploadStats()

            self.reporter(slurm_system, results_dir, bucket="my-bucket").generate()

            store.upload_directory.assert_called_once_with(
                results_dir, f"test_system/current-user/{results_dir.name}", max_workers=8
            )

    @pytest.mark.parametrize("error", [OSError, KeyError, ImportError])
    def test_username_lookup_failure_uploads_under_unknown(
        self, slurm_system: SlurmSystem, results_dir: Path, error: type[Exception]
    ) -> None:
        (results_dir / "experiment.json").unlink()
        with (
            patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls,
            patch("getpass.getuser", side_effect=error("Username unavailable")),
        ):
            store = mock_store_cls.return_value
            store.upload_directory.return_value = UploadStats()

            self.reporter(slurm_system, results_dir, bucket="my-bucket").generate()

            store.upload_directory.assert_called_once_with(
                results_dir, f"test_system/unknown/{results_dir.name}", max_workers=8
            )

    def test_upload_concurrency_is_configurable(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            store = mock_store_cls.return_value
            store.upload_directory.return_value = UploadStats()

            self.reporter(slurm_system, results_dir, bucket="my-bucket", upload_concurrency=16).generate()

            _, kwargs = store.upload_directory.call_args
            assert kwargs["max_workers"] == 16

    def test_no_bucket_uploads_nothing(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            self.reporter(slurm_system, results_dir).generate()

            mock_store_cls.assert_not_called()

    def test_missing_results_dir_uploads_nothing(self, slurm_system: SlurmSystem, tmp_path: Path) -> None:
        reporter = self.reporter(slurm_system, tmp_path / "nope", bucket="my-bucket")
        with patch.object(reporter, "prepare_upload") as prepare:
            reporter.generate()
        prepare.assert_not_called()

    def test_missing_bucket_uploads_nothing(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            store = mock_store_cls.return_value
            store.bucket_exists.return_value = False

            self.reporter(slurm_system, results_dir, bucket="my-bucket").generate()

            store.upload_directory.assert_not_called()

    def test_upload_tree_disabled(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            store = mock_store_cls.return_value

            self.reporter(
                slurm_system, results_dir, bucket="my-bucket", upload_tree=False, upload_tarball=True
            ).generate()

            store.upload_directory.assert_not_called()
            store.upload_file.assert_called_once()

    def test_tarball_created_when_absent(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        tarball_path = Path(str(results_dir) + ".tgz")
        assert not tarball_path.exists()

        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            store = mock_store_cls.return_value
            store.upload_directory.return_value = UploadStats()

            self.reporter(slurm_system, results_dir, bucket="my-bucket", upload_tarball=True).generate()

            assert tarball_path.exists(), "TarballReporter only tarballs on failure, so it must be created here"
            store.upload_file.assert_called_once_with(
                tarball_path, "test_system/test-user/nccl-test_2025-04-16_14-27-45/nccl-test_2025-04-16_14-27-45.tgz"
            )

    def test_fresh_tarball_is_reused(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        tarball_path = Path(str(results_dir) + ".tgz")
        tarball_path.write_bytes(b"pre-existing")
        self._set_tarball_newer_than_contents(results_dir, tarball_path)

        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            mock_store_cls.return_value.upload_directory.return_value = UploadStats()

            self.reporter(slurm_system, results_dir, bucket="my-bucket", upload_tarball=True).generate()

            assert tarball_path.read_bytes() == b"pre-existing"
            mock_store_cls.return_value.upload_file.assert_called_once()

    def test_stale_tarball_is_regenerated(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        """A tarball left by an earlier run must not be uploaded in place of the regenerated reports."""
        tarball_path = Path(str(results_dir) + ".tgz")
        tarball_path.write_bytes(b"stale")
        self._set_tarball_newer_than_contents(results_dir, tarball_path)
        (results_dir / "report.html").write_text("<html>regenerated</html>")
        os.utime(tarball_path, (1, 1))  # tarball predates the regenerated report

        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            mock_store_cls.return_value.upload_directory.return_value = UploadStats()

            self.reporter(slurm_system, results_dir, bucket="my-bucket", upload_tarball=True).generate()

        assert tarball_path.read_bytes() != b"stale"
        with tarfile.open(tarball_path) as tar:
            member = tar.extractfile(f"{results_dir.name}/report.html")
            assert member is not None
            assert member.read() == b"<html>regenerated</html>"

    @staticmethod
    def _set_tarball_newer_than_contents(results_dir: Path, tarball_path: Path) -> None:
        newest = max(p.stat().st_mtime for p in [results_dir, *results_dir.rglob("*")])
        os.utime(tarball_path, (newest + 10, newest + 10))

    def test_env_var_fallback(self, slurm_system: SlurmSystem, results_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv("CLOUDAI_S3_BUCKET", "env-bucket")
        monkeypatch.setenv("CLOUDAI_S3_PREFIX", "env-prefix")
        monkeypatch.setenv("CLOUDAI_S3_ENDPOINT_URL", "http://localhost:9000")

        config = S3UploadConfig(enable=True)

        assert config.bucket == "env-bucket"
        assert config.prefix == "env-prefix"
        assert config.endpoint_url == "http://localhost:9000"

    def test_toml_overrides_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("CLOUDAI_S3_BUCKET", "env-bucket")

        assert S3UploadConfig(enable=True, bucket="toml-bucket").bucket == "toml-bucket"

    def test_upload_failure_does_not_raise(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        with patch("cloudai.s3_reporter.S3ObjectStore") as mock_store_cls:
            store = mock_store_cls.return_value
            store.upload_directory.return_value = UploadStats(failures=[(results_dir / "report.html", "denied")])

            self.reporter(slurm_system, results_dir, bucket="my-bucket").generate()

    def test_requires_at_least_one_upload_mode(self) -> None:
        with pytest.raises(ValidationError, match="upload_tree"):
            S3UploadConfig(enable=True, bucket="my-bucket", upload_tree=False, upload_tarball=False)

    @pytest.mark.parametrize("concurrency", [0, -1])
    def test_upload_concurrency_must_be_positive(self, concurrency: int) -> None:
        with pytest.raises(ValidationError):
            S3UploadConfig(enable=True, bucket="my-bucket", upload_concurrency=concurrency)

    def test_preparation_approves_destination_before_s3_io(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        events = []
        approved_store = MagicMock()
        approved_store.bucket_exists.side_effect = lambda: events.append("bucket-check") or True
        approved_store.upload_directory.return_value = UploadStats()

        class CompanyReporter(S3UploadReporter):
            def prepare_upload(self, request: S3UploadRequest) -> PreparedS3Upload:
                events.append(request)
                return PreparedS3Upload(store=approved_store, key_prefix="approved/run")

        with patch("cloudai.s3_reporter.S3ObjectStore") as default_store:
            CompanyReporter(
                slurm_system,
                TestScenario(name="dummy", test_runs=[]),
                results_dir,
                S3UploadConfig(
                    enable=True,
                    bucket="requested-bucket",
                    prefix="requested",
                    endpoint_url="https://storage.example.org",
                    region="example-region",
                    upload_tarball=True,
                ),
            ).generate()

        assert events == [
            S3UploadRequest(
                bucket="requested-bucket",
                key_prefix=f"requested/test_system/test-user/{results_dir.name}",
                region="example-region",
                endpoint_url="https://storage.example.org",
                results_root=results_dir,
            ),
            "bucket-check",
        ]
        default_store.assert_not_called()
        approved_store.upload_directory.assert_called_once_with(results_dir, "approved/run", max_workers=8)
        approved_store.upload_file.assert_called_once_with(
            Path(str(results_dir) + ".tgz"), f"approved/run/{results_dir.name}.tgz"
        )

    @pytest.mark.parametrize("enabled", [True, False])
    def test_registered_preparation_uses_scenario_config(
        self, slurm_system: SlurmSystem, results_dir: Path, enabled: bool
    ) -> None:
        requests = []
        store = MagicMock()
        store.upload_directory.return_value = UploadStats()

        class CompanyReporter(S3UploadReporter):
            def prepare_upload(self, request: S3UploadRequest) -> PreparedS3Upload:
                requests.append(request)
                return PreparedS3Upload(store=store, key_prefix=request.key_prefix)

        registry = Registry()
        with (
            patch.dict(registry.scenario_reports, {}, clear=True),
            patch.dict(registry.report_configs, {}, clear=True),
            patch("cloudai.s3_reporter.S3ObjectStore") as default_store,
        ):
            registry.update_scenario_report(
                "s3", CompanyReporter, S3UploadConfig(enable=True, bucket="registry-bucket")
            )
            scenario = TestScenario(
                name="dummy", test_runs=[], reports={"s3": S3UploadConfig(enable=enabled, bucket="", prefix="scenario")}
            )
            generate_reports(slurm_system, scenario, results_dir)

        default_store.assert_not_called()
        assert len(requests) == int(enabled)
        if enabled:
            assert requests[0].bucket == ""
            assert requests[0].key_prefix == f"scenario/test_system/test-user/{results_dir.name}"
            store.upload_directory.assert_called_once_with(results_dir, requests[0].key_prefix, max_workers=8)
        else:
            store.upload_directory.assert_not_called()

    def test_preparation_can_skip_upload(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        reporter = self.reporter(slurm_system, results_dir, bucket="my-bucket")
        with (
            patch.object(reporter, "prepare_upload", return_value=None),
            patch("cloudai.s3_reporter.S3ObjectStore") as store,
        ):
            reporter.generate()
        store.assert_not_called()

    def test_preparation_failure_does_not_fall_back(self, slurm_system: SlurmSystem, results_dir: Path) -> None:
        reporter = self.reporter(slurm_system, results_dir, bucket="my-bucket")
        with (
            patch.object(reporter, "prepare_upload", side_effect=RuntimeError("Authorization denied")),
            patch("cloudai.s3_reporter.S3ObjectStore") as store,
            pytest.raises(RuntimeError, match="Authorization denied"),
        ):
            reporter.generate()
        store.assert_not_called()


def test_s3_upload_runs_after_tarball() -> None:
    order = [name for name, _ in Registry().ordered_scenario_reports()]

    assert order.index("s3") > order.index("tarball")
    assert order[-1] == "s3"
