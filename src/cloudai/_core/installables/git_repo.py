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

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict

from .base import Installable, InstallStatusResult

if TYPE_CHECKING:
    from ..base_installer import BaseInstaller


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

    def check_submodules_state(self, repo_path: Path) -> tuple[bool, str]:
        """Check if submodules state in the cloned repo matches self.init_submodules."""
        try:
            result = subprocess.run(
                ["git", "submodule", "status", "--recursive"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
            )
        except OSError as e:
            return False, f"Failed to get submodule status: {e}"
        if result.returncode != 0:
            return False, f"Failed to get submodule status: {result.stderr}"
        output = [line for line in result.stdout.splitlines() if line.strip()]

        has_submodules = bool(output)
        if not has_submodules:
            return True, ""

        status_prefixes = [line[0] for line in output]
        if self.init_submodules and not all(prefix == " " for prefix in status_prefixes):
            return False, "Cloned repo has not all submodules initialized."
        if not self.init_submodules and not all(prefix == "-" for prefix in status_prefixes):
            return False, "Cloned repo has some submodules initialized but requires none to be."

        return True, ""

    def ensure_submodules_state(self, repo_path: Path) -> tuple[bool, str]:
        """Ensure submodules state in the cloned repo matches self.init_submodules (install or deinstall them)."""
        submodules_are_ok, submodules_are_ok_msg = self.check_submodules_state(repo_path)
        if submodules_are_ok:
            return True, ""
        if not submodules_are_ok and "Failed to get submodule status" in submodules_are_ok_msg:
            return False, submodules_are_ok_msg

        cmd = ["update", "--init", "--recursive"] if self.init_submodules else ["deinit", "--all", "--force"]
        action = "initialize" if self.init_submodules else "deinitialize"
        try:
            result = subprocess.run(["git", "submodule", *cmd], cwd=str(repo_path), capture_output=True, text=True)
        except OSError as e:
            return False, f"Failed to {action} submodules: {e}"
        if result.returncode != 0:
            return False, f"Failed to {action} submodules: {result.stderr}"

        return True, ""

    def install(self, installer: "BaseInstaller") -> InstallStatusResult:
        repo_path = installer.system.install_path / self.repo_name
        if repo_path.exists():
            verify_res = self._verify_commit(self.commit, repo_path)
            if not verify_res.success:
                return verify_res
            submodules_res, submodules_msg = self.ensure_submodules_state(repo_path)
            if not submodules_res:
                return InstallStatusResult(False, submodules_msg)
            self.installed_path = repo_path
            msg = f"Git repository already exists at {repo_path}."
            logging.debug(msg)
            return InstallStatusResult(True, msg)

        res = self._clone_and_setup_repo(installer, repo_path)
        if not res.success:
            return res

        self.installed_path = repo_path
        return InstallStatusResult(True)

    def uninstall(self, installer: "BaseInstaller") -> InstallStatusResult:
        logging.debug(f"Uninstalling git repository at {self.installed_path=}")
        repo_path = self.installed_path if self.installed_path else installer.system.install_path / self.repo_name
        if not repo_path.exists():
            return InstallStatusResult(True, f"Repository {self.url} is not cloned.")

        logging.debug(f"Removing folder {repo_path}")
        shutil.rmtree(repo_path)
        self.installed_path = None

        return InstallStatusResult(True)

    def is_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        repo_path = installer.system.install_path / self.repo_name
        if not repo_path.exists():
            return InstallStatusResult(False, f"Git repository {self.url} not cloned")
        verify_res = self._verify_commit(self.commit, repo_path)
        if not verify_res.success:
            return verify_res

        verify_submodules, msg_submodules = self.check_submodules_state(repo_path)
        if not verify_submodules:
            return InstallStatusResult(False, msg_submodules)

        self.installed_path = repo_path
        return InstallStatusResult(True)

    def mark_as_installed(self, installer: "BaseInstaller") -> InstallStatusResult:
        self.installed_path = installer.system.install_path / self.repo_name
        return InstallStatusResult(True)

    def _clone_and_setup_repo(self, installer: "BaseInstaller", repo_path: Path) -> InstallStatusResult:
        res = self._clone_repository(installer, repo_path)
        if not res.success:
            return res

        res = self._checkout_commit(self.commit, repo_path)
        if not res.success:
            logging.error(f"Checkout failed, removing cloned repository at {repo_path}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return res

        submodules_res, submodules_msg = self.ensure_submodules_state(repo_path)
        if not submodules_res:
            logging.error(f"Submodule setup failed with `{submodules_msg}`, removing cloned repository at {repo_path}")
            if repo_path.exists():
                shutil.rmtree(repo_path)
            return InstallStatusResult(False, submodules_msg)

        return InstallStatusResult(True)

    def _clone_repository(self, installer: "BaseInstaller", path: Path) -> InstallStatusResult:
        logging.debug(f"Cloning repository {self.url} into {path}")
        clone_cmd = ["git", "clone"]

        if installer.is_low_thread_environment:
            clone_cmd.extend(["-c", "pack.threads=4"])

        clone_cmd.extend([self.url, str(path)])

        logging.debug(f"Running git clone command: {' '.join(clone_cmd)}")
        try:
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to clone repository: {e}")
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to clone repository: {result.stderr}")
        return InstallStatusResult(True)

    def _checkout_commit(self, commit_hash: str, path: Path) -> InstallStatusResult:
        logging.debug(f"Checking out specific commit in {path}: {commit_hash}")
        try:
            result = subprocess.run(["git", "checkout", commit_hash], cwd=str(path), capture_output=True, text=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to checkout commit {commit_hash}: {e}")
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to checkout commit {commit_hash}: {result.stderr}")
        return InstallStatusResult(True)

    def _verify_commit(self, ref: str, path: Path) -> InstallStatusResult:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path), capture_output=True, text=True)
        except OSError as e:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {e}")
        if result.returncode != 0:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {result.stderr}")
        actual_commit = result.stdout.strip()

        try:
            commit_resolved = subprocess.run(
                ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
        except OSError as e:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {e}")
        if commit_resolved.returncode != 0:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {commit_resolved.stderr}")
        expected_commit = commit_resolved.stdout.strip()

        try:
            branch_resolved = subprocess.run(
                ["git", "symbolic-ref", "--short", "-q", "HEAD"],
                cwd=str(path),
                capture_output=True,
                text=True,
            )
        except OSError as e:
            return InstallStatusResult(False, f"Failed to verify commit in {path}: {e}")
        actual_branch = branch_resolved.stdout.strip() if branch_resolved.returncode == 0 else ""

        if actual_commit == expected_commit or ref == actual_branch:
            return InstallStatusResult(True)

        return InstallStatusResult(
            success=False,
            message=(
                f"Failed to verify commit in {path}: {actual_commit=}, {actual_branch=}, expected was {ref} or "
                f"{expected_commit=}"
            ),
        )
