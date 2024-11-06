from typing import Optional

from cloudai import CmdArgs, Installable, TestDefinition
from cloudai.installer.installables import DockerImage, GitRepo


class SlurmContainerCmdArgs(CmdArgs):
    docker_image_url: str
    repository_url: str
    repository_commit_hash: str


class SlurmContainerTestDefinition(TestDefinition):
    cmd_args: SlurmContainerCmdArgs

    _docker_image: Optional[DockerImage] = None
    _git_repo: Optional[GitRepo] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def git_repo(self) -> GitRepo:
        if not self._git_repo:
            self._python_executable = GitRepo(
                git_url=self.cmd_args.repository_url, commit_hash=self.cmd_args.repository_commit_hash
            )

        return self._python_executable

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image, self.git_repo]

    @property
    def extra_args_str(self) -> str:
        parts = []
        for k, v in self.extra_cmd_args.items():
            parts.append(f"{k} {v}" if v else k)
        return " ".join(parts)
