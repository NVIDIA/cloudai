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

import argparse
import hashlib
import shlex
import socket
import subprocess
from pathlib import Path


def run(command: list[str], dry_run: bool, cwd: Path) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a CloudAI checkout to a cluster and install it with uv.")
    parser.add_argument("host", help="SSH-config alias; remote uv must be on PATH or in ~/.local/bin")
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--include", action="append", default=[], help="Additional checkout-relative path; repeatable")
    parser.add_argument("--extra", action="append", default=[], help="CloudAI dependency extra; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without connecting to the cluster")
    args = parser.parse_args()

    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=args.checkout,
            text=True,
        ).strip()
    ).resolve()
    worktrees = subprocess.check_output(["git", "worktree", "list", "--porcelain", "-z"], cwd=root, text=True)
    main_checkout = Path(worktrees.split("\0")[0].removeprefix("worktree ")).resolve()
    checkout_id = hashlib.sha256(f"{socket.gethostname()}\0{root}".encode()).hexdigest()[:16]
    directory = "cloudai" if root == main_checkout else f"cloudai-worktrees/{checkout_id}"

    paths = ["src", "conf", "pyproject.toml", "uv.lock", "README.md", "LICENSE.md", "pdm_build.py"]
    paths = [path for path in paths if (root / path).exists()] + args.include
    run(["ssh", "--", args.host, f'mkdir -p "$HOME/{directory}"'], args.dry_run, root)
    transfer = ["rsync", "-aR"]
    for pattern in (
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
        ".DS_Store",
        ".*cache*",
        "__pycache__",
        "*.py[cod]",
        "*.egg-info",
    ):
        transfer.extend(["--exclude", pattern])
    run([*transfer, "--", *paths, f"{args.host}:~/{directory}/"], args.dry_run, root)

    install = ["uv", "sync", "--locked"]
    for extra in args.extra:
        install.extend(["--extra", extra])
    commands = [
        f'cd "$HOME/{directory}"',
        'export PATH="$HOME/.local/bin:$PATH"',
        "unset VIRTUAL_ENV",
        'export UV_PROJECT_ENVIRONMENT="$PWD/.venv"',
        shlex.join(install),
        "touch .cloudai-last-used",
    ]
    run(["ssh", "--", args.host, " && ".join(commands)], args.dry_run, root)


if __name__ == "__main__":
    main()
