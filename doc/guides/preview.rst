Inspect commands before execution
=================================

Validate configuration and inspect generated workload commands before submitting work.

Generate commands
-----------------

After preparing ``slurm.toml``, run:

.. code-block:: bash

   uv run cloudai dry-run --system-config slurm.toml \
       --test-scenario conf/experimental/quickstart/nccl.toml

Review the generated scripts and paths reported by CloudAI. Check allocation size, container image, mounts,
workload arguments, and output location before replacing ``dry-run`` with ``run``.

Validate from Python
--------------------

.. code-block:: python

   from pathlib import Path
   import cloudai.api

   cloudai.api.validate_scenario(
       Path("conf/experimental/quickstart/nccl.toml"), Path("slurm.toml")
   )

Validation loads configuration without creating a runner, installing dependencies, or submitting jobs.
It does not check cluster availability or guarantee successful workload execution.
See :doc:`/api` for exception behavior and :doc:`/reference/cli` for CLI options.
