# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from cloudai import Installable


@dataclass
class DockerImage(Installable):
    """Docker image object."""

    url: str
    _installed_path: Optional[Union[str, Path]] = None

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return isinstance(other, DockerImage) and other.url == self.url

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash(self.url)

    def __str__(self) -> str:
        """Return the string representation of the docker image."""
        return f"DockerImage(url={self.url})"

    @property
    def cache_filename(self) -> str:
        """Return the cache filename for the docker image."""
        tag, wo_prefix = "notag", self.url
        is_local = wo_prefix.startswith("/") or wo_prefix.startswith(".")
        if "://" in wo_prefix:
            wo_prefix = self.url.split("://", maxsplit=1)[1]
        if ":" in wo_prefix:
            tag = wo_prefix.rsplit(":", maxsplit=1)[1]
        wo_tag = wo_prefix.rsplit(":", maxsplit=1)[0]
        if is_local:
            img_name = wo_tag.rsplit("/", maxsplit=1)[1]
        else:
            parts = wo_tag.split("/")
            img_name = "_".join(parts[:-1]) + "__" + parts[-1]

        return f"{img_name}__{tag}.sqsh"

    @property
    def installed_path(self) -> Union[str, Path]:
        """Return the cached path or URL of the docker image."""
        if self._installed_path:
            return self._installed_path
        return self.url

    @installed_path.setter
    def installed_path(self, value: Union[str, Path]) -> None:
        self._installed_path = value


@dataclass
class GitRepo(Installable):
    """Git repository object."""

    git_url: str
    commit_hash: str
    _installed_path: Optional[Path] = None

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return isinstance(other, GitRepo) and other.git_url == self.git_url and other.commit_hash == self.commit_hash

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash((self.git_url, self.commit_hash))

    @property
    def repo_name(self) -> str:
        repo_name = self.git_url.rsplit("/", maxsplit=1)[1].replace(".git", "")
        return f"{repo_name}__{self.commit_hash}"

    @property
    def installed_path(self) -> Optional[Path]:
        return self._installed_path

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value


@dataclass
class PythonExecutable(Installable):
    """Python executable object."""

    git_repo: GitRepo
    _venv_path: Optional[Path] = None

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return (
            isinstance(other, PythonExecutable)
            and other.git_repo.git_url == self.git_repo.git_url
            and other.git_repo.commit_hash == self.git_repo.commit_hash
        )

    def __hash__(self) -> int:
        """Hash the installable object."""
        return self.git_repo.__hash__()

    def __str__(self) -> str:
        """Return the string representation of the python executable."""
        return f"PythonExecutable(git_url={self.git_repo.git_url}, commit_hash={self.git_repo.commit_hash})"

    @property
    def venv_name(self) -> str:
        return f"{self.git_repo.repo_name}-venv"

    @property
    def venv_path(self) -> Path:
        if self._venv_path:
            return self._venv_path
        return Path(self.venv_name)

    @venv_path.setter
    def venv_path(self, value: Path) -> None:
        self._venv_path = value
