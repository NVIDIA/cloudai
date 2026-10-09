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

import getpass
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from typing_extensions import Self

from .core import Reporter
from .models.output import Experiment
from .models.scenario import ReportConfig
from .reporter import TarballReporter
from .util.object_store import S3ObjectStore, join_key


class S3UploadConfig(ReportConfig):
    """
    Configuration for uploading a scenario results directory to object storage.

    Destination fields fall back to environment variables when not set in TOML, so a
    cluster-wide destination can be supplied by the environment while a scenario can
    still override it. Credentials are never read from here; boto3 resolves them from
    its standard chain.
    """

    bucket: str = Field(default_factory=lambda: os.getenv("CLOUDAI_S3_BUCKET", ""))
    prefix: str = Field(default_factory=lambda: os.getenv("CLOUDAI_S3_PREFIX", ""))
    endpoint_url: Optional[str] = Field(default_factory=lambda: os.getenv("CLOUDAI_S3_ENDPOINT_URL") or None)
    region: Optional[str] = None
    upload_tree: bool = True
    upload_tarball: bool = False
    upload_concurrency: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def at_least_one_upload_mode(self) -> Self:
        if not (self.upload_tree or self.upload_tarball):
            raise ValueError("At least one of 'upload_tree' or 'upload_tarball' must be enabled.")
        return self


class S3UploadReporter(Reporter):
    """Uploads the scenario results directory to object storage."""

    def generate(self) -> None:
        config = self.config
        if not isinstance(config, S3UploadConfig):
            logging.warning(f"Expected S3UploadConfig, got {type(config).__name__}, skipping results upload.")
            return

        if not config.bucket:
            logging.warning(
                "S3 upload is enabled but no bucket is configured. "
                "Set 'bucket' in the report config or the CLOUDAI_S3_BUCKET environment variable."
            )
            return

        if not self.results_root.exists():
            logging.warning(f"Results directory {self.results_root} does not exist, skipping results upload.")
            return

        store = S3ObjectStore(bucket=config.bucket, endpoint_url=config.endpoint_url, region=config.region)
        if not store.bucket_exists():
            logging.warning(f"Bucket '{config.bucket}' does not exist, skipping results upload.")
            return

        # generate-report has no in-memory experiment, so read the saved snapshot for its owner.
        try:
            user = Experiment.model_validate_json((self.results_root / "experiment.json").read_text()).user
        except (OSError, ValueError) as exc:
            logging.warning("Cannot read experiment owner; falling back to the current user in the S3 path: %s", exc)
            user = ""
        if not user:
            user = getpass.getuser()
        key_prefix = join_key(config.prefix, self.system.name, user, self.results_root.name)

        if config.upload_tree:
            stats = store.upload_directory(self.results_root, key_prefix, max_workers=config.upload_concurrency)
            logging.info(
                f"Uploaded {stats.files_uploaded} file(s), {stats.bytes_uploaded} byte(s) to {store.uri(key_prefix)} "
                f"in {stats.duration_seconds:.2f}s"
            )
            if stats.failures:
                logging.warning(
                    f"Failed to upload {len(stats.failures)} file(s) to {store.uri(key_prefix)}, "
                    "see debug log for details"
                )

        if config.upload_tarball:
            self.upload_tarball(store, key_prefix)

    def upload_tarball(self, store: S3ObjectStore, key_prefix: str) -> None:
        """
        Upload a tarball of the results directory, (re)creating it if it is missing or stale.

        TarballReporter only produces a tarball when a test run failed, so we cannot
        assume one is already present. A leftover tarball from an earlier run may predate
        regenerated reports, so it is reused only if nothing in the directory is newer.
        """
        tarball_path = Path(str(self.results_root) + ".tgz")
        if not tarball_path.exists() or self._tarball_is_stale(tarball_path):
            TarballReporter(self.system, self.test_scenario, self.results_root, self.config).create_tarball(
                self.results_root
            )

        key = join_key(key_prefix, tarball_path.name)
        try:
            store.upload_file(tarball_path, key)
        except Exception as e:
            logging.warning(f"Failed to upload tarball to {store.uri(key)}, see debug log for details")
            logging.debug(e, exc_info=True)
            return
        logging.info(f"Uploaded tarball to {store.uri(key)}")

    def _tarball_is_stale(self, tarball_path: Path) -> bool:
        """Whether anything in the results directory was modified after the tarball was written."""
        tarball_mtime = tarball_path.stat().st_mtime
        entries = [self.results_root, *self.results_root.rglob("*")]
        return any(entry.stat().st_mtime > tarball_mtime for entry in entries)
