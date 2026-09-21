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

"""Generic object storage interface and an S3 implementation."""

from __future__ import annotations

import fnmatch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .lazy_imports import lazy


def join_key(*parts: str) -> str:
    """Join object key parts with '/', dropping empties and collapsing separators."""
    cleaned = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/".join(cleaned)


@dataclass
class UploadStats:
    """Summary of an upload operation."""

    files_uploaded: int = 0
    bytes_uploaded: int = 0
    failures: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return not self.failures


class ObjectStore(ABC):
    """Minimal object storage interface used to publish CloudAI artifacts."""

    @abstractmethod
    def uri(self, key: str) -> str:
        """Return a human-readable URI for the given key, for logging."""
        ...

    @abstractmethod
    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a single file to the given key."""
        ...

    def upload_directory(
        self, local_dir: Path, key_prefix: str = "", exclude: Optional[list[str]] = None
    ) -> UploadStats:
        """
        Upload every file under ``local_dir``, preserving relative paths.

        Args:
            local_dir: Directory to walk.
            key_prefix: Key prefix to place the tree under.
            exclude: Glob patterns matched against paths relative to ``local_dir``.
                Matching files are skipped.

        Returns:
            Stats describing what was uploaded. Individual file failures are collected
            rather than raised, so a single bad file does not abort the whole upload.
        """
        stats = UploadStats()
        exclude = exclude or []

        for path in sorted(local_dir.rglob("*")):
            if not path.is_file():
                continue

            relative = path.relative_to(local_dir)
            if any(fnmatch.fnmatch(str(relative), pattern) for pattern in exclude):
                logging.debug(f"Skipping excluded file {relative}")
                continue

            key = join_key(key_prefix, relative.as_posix())
            try:
                size = path.stat().st_size
                self.upload_file(path, key)
            except Exception as e:
                logging.debug(f"Failed to upload {path} to {self.uri(key)}: {e}", exc_info=True)
                stats.failures.append((path, str(e)))
                continue

            stats.files_uploaded += 1
            stats.bytes_uploaded += size

        return stats


class S3ObjectStore(ObjectStore):
    """
    S3-backed object store.

    Credentials are resolved by boto3's standard chain (``AWS_*`` environment variables,
    ``~/.aws/credentials``, instance/IAM roles), so no secrets are read from CloudAI
    configuration files.
    """

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = lazy.boto3.client("s3", endpoint_url=self.endpoint_url, region_name=self.region)
        return self._client

    def uri(self, key: str) -> str:
        return f"s3://{join_key(self.bucket, key)}"

    def upload_file(self, local_path: Path, key: str) -> None:
        self.client.upload_file(str(local_path), self.bucket, key)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.ClientError as e:
            if e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
                return False
            raise
        return True

    def bucket_exists(self) -> bool:
        """Return whether the configured bucket exists and is accessible."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except self.client.exceptions.ClientError as e:
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status in (404, 403):
                logging.debug(f"Bucket '{self.bucket}' is not accessible: {e}")
                return False
            raise
        return True
