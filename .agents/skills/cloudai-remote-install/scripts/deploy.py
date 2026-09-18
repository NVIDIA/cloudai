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
import shlex
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("host")
    parser.add_argument("--dry-run", action="store_true")
    actions = parser.add_subparsers(dest="action", required=True)
    run = actions.add_parser("run", add_help=False)
    run.add_argument("command")
    copy = actions.add_parser("copy", add_help=False)
    copy.add_argument("destination")
    copy.add_argument("sources", nargs="+")
    fetch = actions.add_parser("fetch", add_help=False)
    fetch.add_argument("source")
    fetch.add_argument("destination")
    args = parser.parse_args()

    ssh = ["ssh", "-T", "-o", "RemoteCommand=none", "-o", "BatchMode=yes"]
    if args.action == "run":
        command = [*ssh, "--", args.host, args.command]
    elif args.action == "fetch":
        source = shlex.quote(args.source.removeprefix("~/"))
        command = ["rsync", "-a", "-e", shlex.join(ssh), "--", f"{args.host}:{source}", args.destination]
    else:
        command = ["rsync", "-a", "-e", shlex.join(ssh)]
        for pattern in (
            ".git",
            ".venv",
            "venv",
            "env",
            ".env",
            ".cloudai.toml",
            ".cloudai-*",
            ".DS_Store",
            ".*cache*",
            "__pycache__",
            "*.py[cod]",
            "*.egg-info",
            "results/",
            "install/",
        ):
            command.extend(["--exclude", pattern])
        destination = shlex.quote(args.destination.removeprefix("~/"))
        command.extend(["--", *args.sources, f"{args.host}:{destination}"])

    print(shlex.join(command), file=sys.stderr, flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
