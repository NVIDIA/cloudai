from typing import Optional

from cloudai import CmdArgs, Installable, TestDefinition
from cloudai.installer.installables import DockerImage


class SlurmContainerCmdArgs(CmdArgs):
    docker_image_url: str


class SlurmContainerTestDefinition(TestDefinition):
    cmd_args: SlurmContainerCmdArgs

    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]
