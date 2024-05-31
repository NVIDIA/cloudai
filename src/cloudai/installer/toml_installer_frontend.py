import os
import shutil
import toml
from typing import Dict

from cloudai._core.system import System
from cloudai._core.registry import Registry
from cloudai._core.base_installer_frontend import BaseInstallerFrontend

class TomlInstallerFrontend(BaseInstallerFrontend):
    """
    Installer frontend that reads configuration from a TOML file.

    Attributes:
        system (System): The system schema object.
        config_file (str): Path to the TOML configuration file.
    """

    def __init__(self, system: System, config_file: str) -> None:
        """
        Initialize the TomlInstallerFrontend with a system object and a TOML configuration file.

        Args:
            system (System): The system schema object.
            config_file (str): Path to the TOML configuration file.
        """
        super().__init__(system)
        self.config_file = config_file

    def parse_config(self) -> Dict:
        """
        Parse the TOML configuration file.

        Returns:
            Dict: Parsed configuration.
        """
        if not os.path.isfile(self.config_file):
            raise FileNotFoundError(f"The configuration file {self.config_file} does not exist.")

        with open(self.config_file, 'r') as f:
            config = toml.load(f)

        return config

    def validate_and_process_config(self, config: Dict) -> Dict:
        """
        Validate and process the parsed configuration.

        Args:
            config (Dict): Parsed configuration.

        Returns:
            Dict: Processed installation arguments.
        """
        install_path = config.get("install_path")
        if not install_path or not os.path.isabs(install_path):
            raise ValueError("The installation path must be an absolute path.")

        if os.path.exists(install_path):
            if config.get("remove_existing", False):
                try:
                    shutil.rmtree(install_path)
                    os.makedirs(install_path)
                except PermissionError:
                    raise PermissionError("You do not have permission to modify the directory.")
            else:
                if not os.access(install_path, os.W_OK):
                    raise PermissionError("You do not have permission to write to the directory.")
        else:
            try:
                os.makedirs(install_path)
            except PermissionError:
                raise PermissionError("You do not have permission to create the directory.")

        templates = config.get("templates", [])
        if not isinstance(templates, list) or not all(isinstance(t, str) for t in templates):
            raise ValueError("Templates must be a list of strings.")

        docker_cache = config.get("docker_cache", False)
        if not isinstance(docker_cache, bool):
            raise ValueError("Docker cache must be a boolean.")

        has_registry_access = config.get("has_registry_access", True)
        if not isinstance(has_registry_access, bool):
            raise ValueError("Has registry access must be a boolean.")

        tgz_path = config.get("tgz_path", "")
        if not has_registry_access and not os.path.isfile(tgz_path):
            raise FileNotFoundError("The specified Docker images tgz file does not exist.")

        return {
            "templates": templates,
            "docker_cache": docker_cache,
            "install_path": install_path,
            "uninstall": False,
            "already_installed": False,
            "tgz_path": tgz_path
        }

    def get_installation_args(self) -> Dict:
        """
        Get installation arguments by parsing the TOML configuration file.

        Returns:
            Dict: Installation arguments.
        """
        config = self.parse_config()
        return self.validate_and_process_config(config)

def main():
    from cloudai.schema.system.slurm import SlurmNode, SlurmNodeState, SlurmSystem

    nodes = [SlurmNode(name=f"node-0{i}", partition="main", state=SlurmNodeState.UNKNOWN_STATE) for i in range(33, 65)]
    backup_nodes = [
        SlurmNode(name=f"node0{i}", partition="backup", state=SlurmNodeState.UNKNOWN_STATE) for i in range(1, 9)
    ]
    system = SlurmSystem(
        name="test_system",
        install_path="/fake/path",
        output_path="/fake/output",
        default_partition="main",
        partitions={"main": nodes, "backup": backup_nodes},
    )

    config_file = "install_config.toml"
    frontend = TomlInstallerFrontend(system, config_file)
    args = frontend.get_installation_args()

    print(args)

if __name__ == "__main__":
    main()
