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

"""
Deploy a checkout using ordinary, policy-permitted OpenSSH and rsync.

Run `plan` locally before `deploy HOST`. Remote uv must already be available on
PATH or in ~/.local/bin. Git-managed remote clones should be maintained with Git,
not adopted by this snapshot helper. No command submits jobs or deletes deployments.
"""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

STATE = ".cloudai-deployment.json"
HELPER = ".cloudai-deploy.py"
KEEP = ".cloudai-keep"
DEFAULT_PATHS = ("src", "conf", "pyproject.toml", "uv.lock", "README.md", "LICENSE.md", "pdm_build.py")
EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".cache",
    ".tox",
    ".nox",
    "node_modules",
    ".DS_Store",
    STATE,
    HELPER,
    KEEP,
}
SHELL_FILES = (".bashrc", ".bash_profile", ".profile", ".zshrc", ".zprofile")


def git(root: Path, *args: str) -> str:
    """Read Git metadata without changing the checkout."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def signature(path: Path) -> str:
    """Fingerprint file contents and permissions, including executable bits."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"{digest.hexdigest()}:{stat.S_IMODE(path.stat().st_mode):o}"


def safe_path(root: Path, name: str) -> Path:
    """Reject traversal and symlinks before accessing deployment-managed files."""
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Not a relative deployment path: {name}")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Refusing symlink: {path}")
    return path


def selected_files(root: Path, includes: list[str]) -> dict[str, Path]:
    """Select tracked/unignored source files and explicitly requested task extras."""
    names = set(
        git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *DEFAULT_PATHS).split("\0")
    )
    for item in includes:
        path = safe_path(root, item)
        if not path.exists():
            raise ValueError(f"Missing include: {item}")
        names.update(str(p.relative_to(root)) for p in (path.rglob("*") if path.is_dir() else [path]))
    files = {}
    for name in sorted(names - {""}):
        parts = Path(name).parts
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in parts):
            continue
        if Path(name).suffix in {".pyc", ".pyo"}:
            continue
        path = safe_path(root, name)
        if path.is_file():
            files[name] = path
    for required in ("pyproject.toml", "uv.lock"):
        if required not in files:
            raise ValueError(f"Missing {required}; select a CloudAI checkout")
    files[HELPER] = Path(__file__).resolve()
    return files


def snapshot(checkout: Path, includes: list[str]) -> tuple[dict[str, Any], dict[str, Path]]:
    """Identify the checkout independently of its branch or detached-HEAD state."""
    root = Path(git(checkout, "rev-parse", "--show-toplevel").strip()).resolve()
    main = Path(git(root, "worktree", "list", "--porcelain", "-z").split("\0")[0].removeprefix("worktree ")).resolve()
    identity = hashlib.sha256(f"{socket.gethostname()}\0{root}".encode()).hexdigest()[:16]
    files = selected_files(root, includes)
    return {
        "checkout_id": identity,
        "main": root == main,
        "revision": git(root, "rev-parse", "HEAD").strip(),
        "files": {name: signature(path) for name, path in files.items()},
    }, files


def destination(home: Path, record: dict[str, Any]) -> Path:
    """Resolve only the reserved main directory or a direct worktree child."""
    identity = record["checkout_id"]
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{16}", identity):
        raise ValueError("Invalid checkout ID")
    return safe_path(home, "cloudai" if record["main"] else f"cloudai-worktrees/{identity}")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace metadata atomically without following an existing destination link."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    os.replace(stream.name, path)


@contextlib.contextmanager
def deployment_lock(target: Path) -> Iterator[None]:
    """Serialize updates and usage records without leaving stale active locks."""
    lock = safe_path(target.parent, f".{target.name}.deploy.lock")
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_update(target: Path, record: dict[str, Any]) -> None:
    """Refuse unknown directories, other checkouts and remote edits before copying."""
    state = safe_path(target, STATE)
    if state.exists():
        previous = json.loads(state.read_text())
        if (previous["checkout_id"], previous["main"]) != (record["checkout_id"], record["main"]):
            raise ValueError(f"Different checkout owns {target}")
    else:
        if any(target.iterdir()):
            raise ValueError(f"Unmanaged directory: {target}; maintain existing clones with Git")
        previous = {"files": {}}
    for name, expected in previous["files"].items():
        path = safe_path(target, name)
        if not path.is_file() or signature(path) != expected:
            raise ValueError(f"Remote edit: {path}; reconcile it before deploying")
    for name in record["files"]:
        path = safe_path(target, name)
        if path.exists() and name not in previous["files"]:
            raise ValueError(f"Unmanaged file would be overwritten: {path}")


def shell_export(home: Path, config: Path, shell_rc: str) -> tuple[Path, str]:
    """Prepare one idempotent export without replacing an existing user setting."""
    path = safe_path(home, shell_rc)
    old = path.read_text() if path.exists() else ""
    start, end = "# BEGIN CloudAI system config", "# END CloudAI system config"
    if old.count(start) != old.count(end) or old.count(start) > 1:
        raise ValueError(f"Malformed CloudAI export block in {path}")
    remaining = re.sub(rf"{start}\n.*?{end}\n?", "", old, flags=re.DOTALL)
    if re.search(r"^\s*(?:export\s+)?CLOUDAI_SYSTEM_CONFIG=", remaining, flags=re.MULTILINE):
        raise ValueError(f"{path} already sets CLOUDAI_SYSTEM_CONFIG; keep or edit that setting explicitly")
    block = f"{start}\nexport CLOUDAI_SYSTEM_CONFIG={shlex.quote(str(config))}\n{end}\n"
    if start in old:
        return path, re.sub(rf"{start}\n.*?{end}\n?", lambda _: block, old, flags=re.DOTALL)
    return path, old + ("\n" if old and not old.endswith("\n") else "") + block


def configuration(
    home: Path, config_name: str | None, shell_rc: str | None
) -> tuple[dict[str, str], tuple[Path, str] | None]:
    """Use the selected config explicitly; persist it only when requested."""
    env = dict(os.environ)
    name = config_name or env.get("CLOUDAI_SYSTEM_CONFIG")
    if not name:
        raise ValueError("Select an existing remote --system-config or export CLOUDAI_SYSTEM_CONFIG first")
    config = Path(name).expanduser()
    if not config.is_absolute() or not config.is_file():
        raise ValueError(f"System config must be an existing absolute remote path: {config}")
    config = config.resolve()
    env["CLOUDAI_SYSTEM_CONFIG"] = str(config)
    export = None
    if shell_rc:
        if shell_rc not in SHELL_FILES:
            raise ValueError("Choose the user's shell startup file")
        if any(config.is_relative_to(home / folder) for folder in ("cloudai", "cloudai-worktrees")):
            raise ValueError("The default system config must live outside deployment directories")
        export = shell_export(home, config, shell_rc)
    return env, export


def apply_snapshot(stage: Path, home: Path, config_name: str | None, shell_rc: str | None, extras: list[str]) -> Path:
    """Apply a prepared snapshot and install it with uv on the destination host."""
    record = json.loads((stage / STATE).read_text())
    for name, expected in record["files"].items():
        if signature(safe_path(stage, name)) != expected:
            raise ValueError(f"Snapshot changed during transfer: {name}")
    env, export = configuration(home, config_name, shell_rc)
    original_rc = export[0].read_text() if export and export[0].exists() else ""
    target = destination(home, record)
    target.mkdir(parents=True, exist_ok=True)
    # The parent lock does not make a new destination appear owned/nonempty.
    with deployment_lock(target):
        check_update(target, record)
        safe_path(target, ".venv")
        state = target / STATE
        previous = json.loads(state.read_text()) if state.exists() else {"files": {}}
        for name in previous["files"].keys() - record["files"].keys():
            safe_path(target, name).unlink()
        for name in record["files"]:
            path = safe_path(target, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe_path(stage, name), path)
        record.update(last_used=time.time(), ready=False)
        write_json(state, record)
        # Do not inherit an environment belonging to another project/checkout.
        env["UV_PROJECT_ENVIRONMENT"] = str(target / ".venv")
        env.pop("VIRTUAL_ENV", None)
        command = ["uv", "sync", "--locked"]
        for extra in extras:
            command.extend(["--extra", extra])
        subprocess.run(command, cwd=target, env=env, check=True)
        subprocess.run(
            ["uv", "run", "--locked", "--no-sync", "cloudai", "verify-configs", env["CLOUDAI_SYSTEM_CONFIG"]],
            cwd=target,
            env=env,
            check=True,
        )
        if export:
            path = safe_path(home, export[0].name)
            if (path.read_text() if path.exists() else "") != original_rc:
                raise ValueError(f"{path} changed during installation; leaving it untouched")
            path.write_text(export[1])
        record.update(ready=True, deployed_at=time.time())
        write_json(state, record)
    return target


def deploy(args: argparse.Namespace) -> None:
    """Stage selected files with rsync and call the helper over ordinary SSH."""
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", args.host):
        raise ValueError("Use an SSH-config host alias or user@hostname")
    record, files = snapshot(args.checkout, args.include)
    with tempfile.TemporaryDirectory(prefix="cloudai-deploy-") as temporary:
        stage = Path(temporary)
        for name, source in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
        # Fingerprint the copied snapshot, not files that could change mid-copy.
        record["files"] = {name: signature(stage / name) for name in files}
        write_json(stage / STATE, record)
        remote = subprocess.check_output(
            ["ssh", args.host, 'mktemp -d "$HOME/.cloudai-stage-XXXXXXXX"'],
            text=True,
        ).strip()
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+/\.cloudai-stage-[A-Za-z0-9]+", remote) or ".." in Path(remote).parts:
            raise ValueError("Unexpected SSH output for remote staging directory")
        print(f"Remote staging directory: {remote}", flush=True)
        subprocess.run(["rsync", "-a", "--", f"{stage}/", f"{args.host}:{remote}/"], check=True)
        command = ["uv", "run", "--no-project", "--python", ">=3.10", "python", f"{remote}/{HELPER}", "apply", remote]
        for option, value in (("--system-config", args.system_config), ("--shell-rc", args.shell_rc)):
            if value:
                command.extend([option, value])
        for extra in args.extra:
            command.extend(["--extra", extra])
        subprocess.run(["ssh", args.host, 'PATH="$HOME/.local/bin:$PATH" ' + shlex.join(command)], check=True)


def maintenance(home: Path, action: str, directory: Path | None) -> list[dict[str, Any]]:
    """Report observed age or mark usage/protection; never delete a deployment."""
    root = safe_path(home, "cloudai-worktrees")
    if action == "status":
        paths = [home / "cloudai", *(sorted(root.iterdir()) if root.exists() else [])]
    else:
        if directory is None:
            raise ValueError("Select a deployment directory")
        paths = [directory.expanduser().absolute()]
    rows = []
    for path in paths:
        if path.is_symlink() or not (path / STATE).is_file():
            if action != "status":
                raise ValueError(f"Not a managed deployment: {path}")
            continue
        record = json.loads(safe_path(path, STATE).read_text())
        if path != destination(home, record):
            raise ValueError(f"Deployment identity does not match directory: {path}")
        if action != "status":
            with deployment_lock(path):
                record = json.loads(safe_path(path, STATE).read_text())
                if action == "mark-used":
                    record["last_used"] = time.time()
                    write_json(path / STATE, record)
                else:
                    safe_path(path, KEEP).touch()
        last_used = record.get("last_used")
        days = (time.time() - last_used) / 86400 if last_used is not None else None
        protected = record["main"] or (path / KEEP).exists()
        rows.append(
            {
                "directory": str(path),
                "ready": record.get("ready", False),
                "protected": protected,
                "days_since_observed_use": days,
                "review_cleanup": not protected and days is not None and days > 21,
            }
        )
    return rows


def main() -> None:
    """Expose planning, deployment and explicitly invoked remote maintenance."""
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Maintenance commands run on the cluster, not over SSH."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "deploy"):
        command = commands.add_parser(
            name, help="Inspect the local snapshot" if name == "plan" else "Deploy over SSH/rsync"
        )
        command.add_argument("--checkout", type=Path, default=Path.cwd())
        command.add_argument(
            "--include", action="append", default=[], help="Additional checkout-relative file/directory; repeatable"
        )
        if name == "deploy":
            command.add_argument("host", help="SSH-config alias; access must already be permitted")
    remote = commands.add_parser("apply", help="Apply a transferred snapshot on the remote host")
    remote.add_argument("stage", type=Path)
    for command in (commands.choices["deploy"], remote):
        command.add_argument(
            "--system-config", help="Existing absolute remote config; defaults to remote CLOUDAI_SYSTEM_CONFIG"
        )
        command.add_argument(
            "--shell-rc", choices=SHELL_FILES, help="Explicitly persist the default config in this remote file"
        )
        command.add_argument(
            "--extra", action="append", default=[], help="Optional CloudAI dependency extra; repeatable"
        )
    commands.add_parser("status", help="List remote installations and cleanup candidates; does not inspect jobs")
    for name in ("mark-used", "protect"):
        command = commands.add_parser(
            name, help="Record use" if name == "mark-used" else "Do not suggest this directory for cleanup"
        )
        command.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            record, _ = snapshot(args.checkout, args.include)
            record["destination"] = "~/cloudai" if record["main"] else f"~/cloudai-worktrees/{record['checkout_id']}"
            print(json.dumps(record, indent=2))
        elif args.command == "deploy":
            deploy(args)
        elif args.command == "apply":
            stage = args.stage.absolute()
            if (
                stage.parent != Path.home()
                or not re.fullmatch(r"\.cloudai-stage-[A-Za-z0-9]+", stage.name)
                or stage.is_symlink()
            ):
                raise ValueError("Apply requires a staging directory created by deploy in remote home")
            print(apply_snapshot(stage, Path.home(), args.system_config, args.shell_rc, args.extra))
            shutil.rmtree(stage)
        else:
            print(json.dumps(maintenance(Path.home(), args.command, getattr(args, "directory", None)), indent=2))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
