import os
import shutil
from typing import List

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.shortcuts import input_dialog, message_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import CheckboxList

from cloudai._core.base_installer_frontend import BaseInstallerFrontend
from cloudai._core.registry import Registry
from cloudai._core.system import System


class UserInteractiveInstallerFrontend(BaseInstallerFrontend):
    """
    User-interactive installer frontend that provides a full-screen UI using the prompt_toolkit library.

    Attributes
        system (System): The system schema object.
        selected_templates (List[str]): List of selected templates.
    """

    def __init__(self, system: System) -> None:
        """
        Initialize the UserInteractiveInstallerFrontend with a system object.

        Args:
            system (System): The system schema object.
        """
        super().__init__(system)
        self.selected_templates = []

    def display_welcome_message(self) -> None:
        """Display a welcome message to the user."""
        message_dialog(
            title="CloudAI Interactive Shell",
            text=(
                "Welcome to the CloudAI Interactive Shell for installation. "
                "This tool will guide you through the installation process. "
                "Press Enter to proceed to the next step."
            ),
        ).run()

    def display_template_selection(self) -> List[str]:
        """
        Display the test template selection screen using a full-screen interactive UI.

        Returns
            List[str]: Selected test template names.
        """
        registry = Registry()
        test_template_names = list(registry.test_templates_map.keys())

        checkbox_list = CheckboxList(values=[(name, name) for name in test_template_names])

        header = Window(
            content=FormattedTextControl("Select test templates to install (Space to select, 'd' to finish):"),
            height=1,
            style="class:header",
        )

        layout = Layout(HSplit([header, checkbox_list]))

        kb = KeyBindings()

        @kb.add("c-q")
        def exit_app(event):
            self.selected_templates = [item for item in checkbox_list.current_values]
            event.app.exit()

        @kb.add("d")
        def submit_selection(event):
            self.selected_templates = [item for item in checkbox_list.current_values]
            event.app.exit()

        app = Application(
            layout=layout, key_bindings=kb, full_screen=True, style=Style.from_dict({"header": "bold underline"})
        )
        app.run()

        return self.selected_templates

    def ask_yes_no_question(self, title: str, text: str) -> str:
        """
        Ask the user a yes-or-no question and validate the input.

        Args:
            title (str): The title of the dialog.
            text (str): The text of the dialog.

        Returns:
            str: The validated yes or no answer.
        """
        while True:
            answer = input_dialog(title=title, text=text).run().strip().lower()
            if answer in ["yes", "no"]:
                return answer

    def ask_installation_directory(self) -> str:
        """
        Ask the user for the installation directory and ensure it is valid.

        Returns
            str: The installation directory path.
        """
        while True:
            install_path = (
                input_dialog(title="Installation Path", text="Enter the absolute path to the installation directory:")
                .run()
                .strip()
            )

            if os.path.isabs(install_path):
                if os.path.exists(install_path):
                    if (
                        self.ask_yes_no_question(
                            title="Directory Exists",
                            text="The directory already exists. Do you want to remove the existing files? (yes/no)",
                        )
                        == "yes"
                    ):
                        try:
                            shutil.rmtree(install_path)
                            os.makedirs(install_path)
                            return install_path
                        except PermissionError:
                            message_dialog(
                                title="Permission Error",
                                text="You do not have permission to modify the directory. Please choose another path.",
                            ).run()
                    else:
                        return install_path
                else:
                    try:
                        os.makedirs(install_path)
                        return install_path
                    except PermissionError:
                        message_dialog(
                            title="Permission Error",
                            text="You do not have permission to create the directory. Please choose another path.",
                        ).run()
            else:
                message_dialog(
                    title="Invalid Path", text="The provided path is not absolute. Please provide an absolute path."
                ).run()

    def ask_docker_cache_option(self) -> dict:
        """
        Ask the user for Docker cache options.

        Returns
            dict: Docker cache options.
        """
        has_registry_access = self.ask_yes_no_question(
            title="Docker Registry Access",
            text="Do you have access to the Docker registries described in the schema? (yes/no)",
        )

        if has_registry_access == "yes":
            docker_cache = self.ask_yes_no_question(
                title="Docker Cache",
                text=(
                    "Do you want to download and cache Docker images locally? "
                    "This is not recommended if enroot caches are large enough and "
                    "the network speed to the registry is fast. (yes/no)"
                ),
            )
            return {"has_registry_access": True, "docker_cache": docker_cache == "yes"}
        else:
            while True:
                tgz_path = (
                    input_dialog(
                        title="Docker Images TGZ", text="Enter the path to the tar-gzipped Docker images file:"
                    )
                    .run()
                    .strip()
                )
                if os.path.isfile(tgz_path):
                    return {"has_registry_access": False, "tgz_path": tgz_path}
                else:
                    message_dialog(
                        title="File Not Found",
                        text="The specified file does not exist. Please provide a valid file path.",
                    ).run()

    def get_installation_args(self) -> dict:
        """
        Get installation arguments from the user through an interactive UI.

        Returns
            dict: Dictionary of installation arguments.
        """
        self.display_welcome_message()

        already_installed = self.ask_yes_no_question(
            title="CloudAI Installation", text="Do you already have a directory where CloudAI is installed? (yes/no)"
        )

        install_path = self.ask_installation_directory()

        if already_installed == "yes":
            return {
                "templates": [],
                "docker_cache": False,
                "install_path": install_path,
                "uninstall": False,
                "already_installed": True,
            }
        else:
            selected_templates = self.display_template_selection()
            docker_options = self.ask_docker_cache_option()
            return {
                "templates": selected_templates,
                "docker_cache": docker_options.get("docker_cache", False),
                "install_path": install_path,
                "uninstall": False,
                "already_installed": False,
                "tgz_path": docker_options.get("tgz_path"),
            }


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
    frontend = UserInteractiveInstallerFrontend(system)
    args = frontend.get_installation_args()

    print(args)


if __name__ == "__main__":
    main()
