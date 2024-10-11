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
class PythonExecutable(Installable):
    """Python executable object."""

    git_url: str
    commit_hash: str
    _installed_path: Optional[Path] = None
    _venv_path: Optional[Path] = None

    def __eq__(self, other: object) -> bool:
        """Check if two installable objects are equal."""
        return (
            isinstance(other, PythonExecutable)
            and other.git_url == self.git_url
            and other.commit_hash == self.commit_hash
        )

    def __hash__(self) -> int:
        """Hash the installable object."""
        return hash((self.git_url, self.commit_hash))

    @property
    def repo_name(self) -> str:
        return self.git_url.rsplit("/", maxsplit=1)[1].replace(".git", "")

    @property
    def venv_name(self) -> str:
        return f"{self.repo_name}-venv"

    @property
    def venv_path(self) -> Optional[Path]:
        return self._venv_path

    @venv_path.setter
    def venv_path(self, value: Path) -> None:
        self._venv_path = value

    @property
    def installed_path(self) -> Optional[Path]:
        return self._installed_path

    @installed_path.setter
    def installed_path(self, value: Path) -> None:
        self._installed_path = value
