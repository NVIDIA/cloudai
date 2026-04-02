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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


class Installable(ABC):
    """Installable object."""

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...


@dataclass
class DockerImage(Installable):
    """Docker image object."""

    url: str
    _installed_path: Optional[Union[str, Path]] = field(default=None, repr=False)

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
        """
        Return the cache filename for the docker image.

        Examples::

            DockerImage("nvcr.io#nvidia/pytorch:24.02-py3").cache_filename
            # "nvcr.io_nvidia__pytorch__24.02-py3.sqsh"

            DockerImage("registry.example.com:5000#group/project").cache_filename
            # "registry.example.com_5000_group__project__notag.sqsh"

            DockerImage("/local/cache/image.sqsh").cache_filename
            # "image.sqsh__notag.sqsh"
        """
        reference = self.url.split("://", maxsplit=1)[-1]
        tag = "notag"

        if reference.startswith("/") or reference.startswith("."):
            # Local image file
            image_ref = reference
            if ":" in reference:
                image_ref, tag = reference.rsplit(":", maxsplit=1)

            # /local/disk/file.sqsh -> file
            image_name = image_ref.rsplit("/", maxsplit=1)[-1]
        else:
            # Remote image url
            parts = reference.replace("#", "/").split("/")

            last_part = parts[-1]
            if ":" in last_part:
                parts[-1], tag = last_part.rsplit(":", maxsplit=1)

            image_name = "_".join(parts[:-1]) + "__" + parts[-1]

        return f"{image_name.replace('#', '_').replace(':', '_')}__{tag}.sqsh"

    @property
    def installed_path(self) -> Union[str, Path]:
        """Return the cached path or URL of the docker image."""
        if self._installed_path:
            return self._installed_path.absolute() if isinstance(self._installed_path, Path) else self._installed_path
        return self.url

    @installed_path.setter
    def installed_path(self, value: Union[str, Path]) -> None:
        self._installed_path = value


class GitRepo(Installable, BaseModel):
    """Git repository object."""

    model_config = ConfigDict(extra="forbid")

    url: str
    commit: str
    init_submodules: bool = False
    installed_path: Optional[Path] = None
    mount_as: Optional[str] = None

    def __repr__(self) -> str:
        return f"GitRepo(url={self.url}, commit={self.commit})"

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return isinstance(other, GitRepo) and other.url == self.url and other.commit == self.commit

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash((self.url, self.commit))

    @property
    def repo_name(self) -> str:
        repo_name = self.url.rsplit("/", maxsplit=1)[1].replace(".git", "")
        return f"{repo_name}__{self.commit}"

    @property
    def container_mount(self) -> str:
        return self.mount_as or f"/git/{self.repo_name}"


@dataclass
class PythonExecutable(Installable):
    """Python executable object."""

    git_repo: GitRepo
    venv_path: Optional[Path] = None
    project_subpath: Optional[Path] = None
    dependencies_from_pyproject: bool = True

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return (
            isinstance(other, PythonExecutable)
            and other.git_repo.url == self.git_repo.url
            and other.git_repo.commit == self.git_repo.commit
        )

    def __hash__(self) -> int:
        """Hash the installable object."""
        return self.git_repo.__hash__()

    def __str__(self) -> str:
        """Return the string representation of the python executable."""
        return f"PythonExecutable(git_url={self.git_repo.url}, commit_hash={self.git_repo.commit})"

    @property
    def venv_name(self) -> str:
        return f"{self.git_repo.repo_name}-venv"


@dataclass
class File(Installable):
    """File object."""

    src: Path
    _installed_path: Optional[Path] = field(default=None, repr=False)

    @property
    def installed_path(self) -> Path:
        return self._installed_path or self.src

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, File) and other.src == self.src

    def __hash__(self) -> int:
        return hash(self.src)


@dataclass
class HFModel(Installable):
    """HuggingFace Model object."""

    model_name: str
    _installed_path: Path | None = field(default=None, repr=False)

    @property
    def installed_path(self) -> Path:
        if self._installed_path:
            return self._installed_path
        return Path("hub") / self.model_name

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HFModel) and other.model_name == self.model_name

    def __hash__(self) -> int:
        return hash(self.model_name)
