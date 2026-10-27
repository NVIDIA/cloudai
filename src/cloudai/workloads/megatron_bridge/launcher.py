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

"""Subprocess-only adapters for the unmodified MBridge performance CLI."""


def main(launcher: str, output: str) -> None:
    """Generate a job or evaluate its completed logs without submitting another job."""
    import importlib
    import json
    import os
    import runpy
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from typing import Any

    run: Any = importlib.import_module("nemo_run")

    root = Path(output)
    context_path = root / "cloudai_megatron_bridge_context.json"
    sys.path.insert(0, str(Path(launcher).parent))
    evaluate = sys.argv[1:2] == ["--cloudai-evaluate"]

    if evaluate:
        wandb = importlib.import_module("wandb")

        context = json.loads(context_path.read_text())
        sys.argv = [launcher, *context["args"]]
        if context["wandb_run_id"]:
            wandb.util.generate_id = lambda: os.environ.get("WANDB_RUN_ID", context["wandb_run_id"])
        submitted = False

        def completed_run(*args, **kwargs):
            nonlocal submitted
            if submitted:
                raise RuntimeError("Evaluation attempted to submit another training job.")
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
        args = sys.argv[1:]
        sys.argv.append("-d")
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
            return original_run(*run_args, **kwargs)

        run.Experiment.dryrun = keep_artifacts
        run.run = generate

    runpy.run_path(launcher, run_name="__main__")
