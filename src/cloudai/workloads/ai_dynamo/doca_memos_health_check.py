# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a small synchronous DOCA_MEMOS write/query/read health check."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import json
import mmap
import os
import re
import stat
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

_NVME_GENERIC_DEVICE = re.compile(r"^ng(?P<controller>\d+)n\d+$")
_NVME_BLOCK_DEVICE = re.compile(r"^nvme(?P<controller>\d+)n\d+$")
_NVME_CONTROLLER = re.compile(r"^nvme(?P<controller>\d+)$")
_RUNTIME_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_AUTO_DEVICE_NAME = "auto"
_NODE_CONFIG_PREFIX = "lmcache-config"
_HUGEPAGE_SIZE = 2 * 1024 * 1024
_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26
_MAP_HUGE_2MB = 21 << _MAP_HUGE_SHIFT


@contextlib.contextmanager
def _hugepage_buffer(size: int) -> Any:
    """Allocate a Linux 2 MiB hugetlb mapping and yield its address."""
    if size <= 0:
        raise ValueError("hugepage buffer size must be positive")

    allocation_size = ((size + _HUGEPAGE_SIZE - 1) // _HUGEPAGE_SIZE) * _HUGEPAGE_SIZE
    libc = ctypes.CDLL(None, use_errno=True)
    mmap_fn = libc.mmap
    mmap_fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    mmap_fn.restype = ctypes.c_void_p
    munmap_fn = libc.munmap
    munmap_fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    munmap_fn.restype = ctypes.c_int

    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | _MAP_HUGETLB | _MAP_HUGE_2MB
    address = mmap_fn(
        None,
        allocation_size,
        mmap.PROT_READ | mmap.PROT_WRITE,
        flags,
        -1,
        0,
    )
    if address == ctypes.c_void_p(-1).value:
        error = ctypes.get_errno()
        raise OSError(error, f"Unable to allocate {allocation_size} bytes from the 2 MiB hugetlb pool")

    print(
        f"DOCA_MEMOS_HUGEPAGE_BUFFER_OK address=0x{address:x} size={size} allocation_size={allocation_size}",
        flush=True,
    )
    try:
        yield address
    finally:
        if munmap_fn(address, allocation_size) != 0:
            error = ctypes.get_errno()
            raise OSError(error, f"Unable to unmap hugetlb buffer at 0x{address:x}")


def _controller_name(device: Path) -> str:
    """Return the sysfs controller name for an NVMe namespace device."""
    for pattern in (_NVME_GENERIC_DEVICE, _NVME_BLOCK_DEVICE):
        match = pattern.fullmatch(device.name)
        if match:
            return f"nvme{match.group('controller')}"
    raise ValueError(f"Cannot derive an NVMe controller from device name {device}")


def discover_doca_memos_device(
    sysfs_root: Path = Path("/sys/class/nvme"),
    dev_root: Path = Path("/dev"),
) -> str:
    """Return the sole generic namespace for a live DOCA/SNAP controller."""
    candidates: list[Path] = []
    for controller_path in sorted(sysfs_root.glob("nvme*"), key=lambda path: path.name):
        match = _NVME_CONTROLLER.fullmatch(controller_path.name)
        if not match:
            continue
        try:
            state = (controller_path / "state").read_text().strip()
            model = (controller_path / "model").read_text().strip()
        except OSError:
            continue
        if state != "live" or ("DOCA" not in model and "SNAP" not in model):
            continue

        controller_index = match.group("controller")
        for device in sorted(dev_root.glob(f"ng{controller_index}n*"), key=lambda path: path.name):
            if _NVME_GENERIC_DEVICE.fullmatch(device.name):
                candidates.append(device)

    if len(candidates) != 1:
        devices = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            "Expected exactly one generic namespace for a live DOCA/SNAP controller, "
            f"found {len(candidates)}: {devices}"
        )

    device = str(candidates[0])
    print(f"DOCA_MEMOS_DEVICE_DISCOVERED device={device}", flush=True)
    return device


def _runtime_identity() -> tuple[str, str]:
    """Return safe Slurm job and node identifiers for a generated config."""
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    node_name = os.environ.get("SLURMD_NODENAME", "").strip() or os.uname().nodename
    for label, value in (("SLURM_JOB_ID", job_id), ("node name", node_name)):
        if not value or not _RUNTIME_COMPONENT.fullmatch(value):
            raise RuntimeError(f"Cannot generate node-local LMCache config: invalid {label} {value!r}")
    return job_id, node_name


def materialize_node_lmcache_config(
    base_config_path: Path,
    output_dir: Path,
    device_name: str,
) -> Path:
    """Write a job- and node-specific LMCache config using the discovered device."""
    config = json.loads(base_config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("LMCache base config must be a JSON object")
    extra_config = config.get("extra_config")
    if not isinstance(extra_config, dict):
        raise ValueError("LMCache base config extra_config must be a mapping")
    backend_params = extra_config.get("nixl_backend_params")
    if backend_params is None:
        backend_params = {}
        extra_config["nixl_backend_params"] = backend_params
    elif not isinstance(backend_params, dict):
        raise ValueError("LMCache base config nixl_backend_params must be a mapping")
    backend_params["device_name"] = device_name

    job_id, node_name = _runtime_identity()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_NODE_CONFIG_PREFIX}-{job_id}-{node_name}.yaml"
    temporary_path = output_dir / f".{output_path.name}.{os.getpid()}.tmp"
    try:
        temporary_path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")
        os.replace(temporary_path, output_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
    print(f"DOCA_MEMOS_LMCACHE_CONFIG_WRITTEN path={output_path} device={device_name}", flush=True)
    return output_path


def validate_device(device_name: str) -> None:
    """Fail unless the configured DOCA namespace and controller look usable."""
    device = Path(device_name)
    device_stat = device.stat()
    if not (stat.S_ISCHR(device_stat.st_mode) or stat.S_ISBLK(device_stat.st_mode)):
        raise RuntimeError(f"{device} is not a character or block device")
    if not os.access(device, os.R_OK | os.W_OK):
        raise PermissionError(f"{device} must be readable and writable")

    controller = _controller_name(device)
    controller_path = Path("/sys/class/nvme") / controller
    state = (controller_path / "state").read_text().strip()
    if state != "live":
        raise RuntimeError(f"{controller} is not live (state={state!r})")

    model = (controller_path / "model").read_text().strip()
    if "DOCA" not in model and "SNAP" not in model:
        raise RuntimeError(f"{controller} model {model!r} is not a DOCA/SNAP controller")
    print(f"DOCA_MEMOS_DEVICE_OK device={device} controller={controller} model={model!r}", flush=True)


def _load_nixl() -> tuple[Any, Any, Any]:
    """Load either the standard or CUDA-versioned NIXL Python package."""
    errors: list[str] = []
    for package in ("nixl", "nixl_cu13"):
        try:
            utils = importlib.import_module(f"{package}._utils")
            api = importlib.import_module(f"{package}._api")
            return utils, api.nixl_agent, api.nixl_agent_config
        except ImportError as exc:
            errors.append(f"{package}: {exc}")
    raise ImportError("Unable to import a NIXL Python package: " + "; ".join(errors))


def _wait_for_transfer(agent: Any, handle: Any, timeout_seconds: float) -> None:
    """Wait for one NIXL transfer and fail on timeout or backend error."""
    state = agent.transfer(handle)
    deadline = time.monotonic() + timeout_seconds
    while state not in {"DONE", "ERR"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"NIXL transfer did not finish within {timeout_seconds:.1f}s")
        time.sleep(0.001)
        state = agent.check_xfer_state(handle)
    if state != "DONE":
        raise RuntimeError(f"NIXL transfer finished in state {state!r}")


def _backend_params(raw_params: str) -> dict[str, str]:
    parsed = json.loads(raw_params)
    if not isinstance(parsed, dict):
        raise ValueError("backend parameters must be a JSON object")
    params = {str(key): str(value) for key, value in parsed.items()}
    params["query_mem_mode"] = "actual"
    params["num_tasks"] = "1"
    return params


def run_health_check(backend_params: dict[str, str], size: int, transfer_timeout: float) -> None:
    """Write, query, and read back one object through DOCA_MEMOS."""
    device_name = backend_params.get("device_name")
    if not device_name:
        raise ValueError("DOCA_MEMOS backend parameters require device_name")
    if size <= 0:
        raise ValueError("probe size must be positive")
    if transfer_timeout <= 0:
        raise ValueError("transfer timeout must be positive")

    validate_device(device_name)
    _, nixl_agent, nixl_agent_config = _load_nixl()
    agent_name = f"CloudAIDocaMemosHealth{os.getpid()}_{time.time_ns()}"
    agent = nixl_agent(
        agent_name,
        nixl_agent_config(
            backends=[],
            enable_prog_thread=True,
            enable_listen_thread=False,
        ),
    )
    agent.create_backend("DOCA_MEMOS", backend_params)

    with _hugepage_buffer(size) as src_addr, _hugepage_buffer(size) as dst_addr:
        ctypes.memset(src_addr, 0xA5, size)
        ctypes.memset(dst_addr, 0, size)

        local_reg = None
        object_reg = None
        handles: list[Any] = []
        key = uuid.uuid4().hex
        try:
            local_reg = agent.register_memory(
                [
                    (src_addr, size, 0, ""),
                    (dst_addr, size, 0, ""),
                ],
                "DRAM",
                backends=["DOCA_MEMOS"],
            )
            object_reg = agent.register_memory(
                [(0, size, 1, key)],
                "OBJ",
                backends=["DOCA_MEMOS"],
            )

            object_xfer = object_reg.trim()
            src_xfer = agent.get_xfer_descs([(src_addr, size, 0)], "DRAM")
            dst_xfer = agent.get_xfer_descs([(dst_addr, size, 0)], "DRAM")

            write_handle = agent.initialize_xfer("WRITE", src_xfer, object_xfer, agent_name)
            handles.append(write_handle)
            started = time.monotonic()
            _wait_for_transfer(agent, write_handle, transfer_timeout)
            print(f"DOCA_MEMOS_WRITE_OK seconds={time.monotonic() - started:.6f}", flush=True)

            started = time.monotonic()
            query = agent.query_memory(
                [(0, 0, 1, key)],
                "DOCA_MEMOS",
                mem_type="OBJ",
            )
            if not query or query[0] is None:
                raise RuntimeError("DOCA_MEMOS query did not find the object just written")
            print(f"DOCA_MEMOS_QUERY_OK seconds={time.monotonic() - started:.6f}", flush=True)

            read_handle = agent.initialize_xfer("READ", dst_xfer, object_xfer, agent_name)
            handles.append(read_handle)
            started = time.monotonic()
            _wait_for_transfer(agent, read_handle, transfer_timeout)
            if ctypes.string_at(src_addr, size) != ctypes.string_at(dst_addr, size):
                raise RuntimeError("DOCA_MEMOS read-back data does not match the write")
            print(f"DOCA_MEMOS_READ_OK seconds={time.monotonic() - started:.6f}", flush=True)
            validate_device(device_name)
        finally:
            for handle in handles:
                with contextlib.suppress(Exception):
                    agent.release_xfer_handle(handle)
            if object_reg is not None:
                with contextlib.suppress(Exception):
                    agent.deregister_memory(object_reg, backends=["DOCA_MEMOS"])
            if local_reg is not None:
                with contextlib.suppress(Exception):
                    agent.deregister_memory(local_reg, backends=["DOCA_MEMOS"])

    print(f"DOCA_MEMOS_HEALTH_CHECK_OK size={size}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-params-json", required=True)
    parser.add_argument("--base-lmcache-config-json", type=Path)
    parser.add_argument("--output-config-dir", type=Path)
    parser.add_argument("--skip-data-path-check", action="store_true")
    parser.add_argument("--size", type=int, default=6 * 1024 * 1024)
    parser.add_argument("--transfer-timeout", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        backend_params = _backend_params(args.backend_params_json)
        device_name = backend_params.get("device_name", "").strip()
        if not device_name or device_name.casefold() == _AUTO_DEVICE_NAME:
            device_name = discover_doca_memos_device()
            backend_params["device_name"] = device_name
            if args.base_lmcache_config_json is None or args.output_config_dir is None:
                raise ValueError(
                    "automatic DOCA_MEMOS discovery requires --base-lmcache-config-json and --output-config-dir"
                )
            materialize_node_lmcache_config(
                args.base_lmcache_config_json,
                args.output_config_dir,
                device_name,
            )

        if args.skip_data_path_check:
            validate_device(device_name)
            print("DOCA_MEMOS_DATA_PATH_CHECK_SKIPPED", flush=True)
        else:
            run_health_check(
                backend_params,
                size=args.size,
                transfer_timeout=args.transfer_timeout,
            )
    except Exception as exc:
        print(f"DOCA_MEMOS_HEALTH_CHECK_FAILED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
