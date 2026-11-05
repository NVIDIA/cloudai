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
import importlib
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _reuse_wandb_run_id(run_id: str) -> None:
    runid: Any = importlib.import_module("wandb.sdk.lib.runid")
    runid.generate_id = lambda: run_id
    wandb = importlib.import_module("wandb")
    if hasattr(wandb.util, "generate_id"):
        wandb.util.generate_id = lambda: run_id


def main(mode: str, launcher: str, output: str, args: list[str]) -> None:
    """Generate a job or evaluate its completed logs without submitting another job."""
    run: Any = importlib.import_module("nemo_run")

    root = Path(output)
    context_path = root / "cloudai_megatron_bridge_context.json"
    sys.path.insert(0, str(Path(launcher).parent))
    sys.argv = [launcher, *args]

    if mode == "evaluate":
        context = json.loads(context_path.read_text())
        sys.argv = [launcher, *context["args"]]
        run_id: str | None = context["wandb_run_id"]
        if run_id:
            run_id = os.environ.get("WANDB_RUN_ID", run_id)
            _reuse_wandb_run_id(run_id)
        submitted = False

        def completed_run(*args, **kwargs):
            nonlocal submitted
            if submitted:
                raise RuntimeError("Evaluation attempted to submit another training job.")
            if run_id and kwargs["executor"].env_vars.get("WANDB_RUN_ID") != run_id:
                raise RuntimeError("Evaluation did not preserve the training wandb run ID.")
            submitted = True
            for key, value in kwargs["executor"].env_vars.items():
                if key.startswith("WANDB_"):
                    os.environ.setdefault(key, value)

        run.run = completed_run
        job_dir = str(next(root.glob("experiments/**/*_sbatch.sh")).parent / context["name"])
        run.Experiment.from_title = lambda *args, **kwargs: SimpleNamespace(
            status=lambda **kwargs: {"completed": {"local_dir": job_dir, "status": "SUCCEEDED"}}
        )
    else:
        original_dryrun = run.Experiment.dryrun
        original_run = run.run

        def keep_artifacts(self, log=True, exist_ok=False, delete_exp_dir=True):
            del delete_exp_dir
            return original_dryrun(self, log=log, exist_ok=exist_ok, delete_exp_dir=False)

        def generate(*run_args, **kwargs):
            context_path.write_text(
                json.dumps(
                    {
                        "args": args,
                        "name": kwargs["name"],
                        "wandb_run_id": kwargs["executor"].env_vars.get("WANDB_RUN_ID"),
                    }
                )
            )
            kwargs["dryrun"] = True
            original_run(*run_args, **kwargs)
            # Stop before the native launcher monitors or evaluates the generated job.
            raise SystemExit(0)

        run.Experiment.dryrun = keep_artifacts
        run.run = generate

    runpy.run_path(launcher, run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "evaluate"])
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--output", required=True)
    options, native_args = parser.parse_known_args()
    main(options.mode, options.launcher, options.output, native_args[1:])
