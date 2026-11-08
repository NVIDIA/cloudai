Run on Slurm
============

The following are subtopics related to Slurm:

Single sbatch vs Per-case sbatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI supports two modes for submitting Slurm jobs:

1. (default) Per-case sbatch mode: each case is submitted as a separate sbatch job, allowing flexible scheduling and dependency management.

   - Suits best when cases can run in parallel or there are no dependencies between cases.

2. Single sbatch mode: all cases are submitted together in a single sbatch job and share the same nodes. Each next case starts after the previous one completes. To enable it one needs to pass ``--single-sbatch`` flag to ``cloudai run`` command (works only for Slurm systems).

   - Suits best for jobs when cases need to run on the same nodes for performance reasons.
   - There is no support for dependency management between cases yet; all jobs run one after another.

Assuming two cases in a scenario like this:

.. code-block:: toml

   [[Tests]]
   id = "nccl.all_reduce"
   test_name = "nccl-all_reduce"
   num_nodes = 2
   time_limit = "00:20:00"

   [[Tests]]
   id = "nccl.all_gather"
   test_name = "nccl-all_gather"
   num_nodes = 2
   time_limit = "00:20:00"

Regular output directory structure (some files are omitted for clarity):

.. code-block:: bash

   $ tree results/scenario
   results/scenario
   ├── nccl.all_gather
   │   └── 0
   │       ├── cloudai_sbatch_script.sh
   │       ├── env_vars.sh
   │       ├── metadata/
   │       ├── stderr.txt
   │       ├── stdout.txt
   │       └── test-run.toml
   └── nccl.all_reduce
       └── 0
           ├── cloudai_sbatch_script.sh
           ├── env_vars.sh
           ├── metadata/
           ├── stderr.txt
           ├── stdout.txt
           └── test-run.toml

Single sbatch mode output directory structure:

.. code-block:: bash

   $ tree results/scenario
   results/scenario
   ├── cloudai_sbatch_script.sh
   ├── common.err
   ├── common.out
   ├── metadata/
   ├── nccl.all_gather
   │   └── 0
   │       ├── env_vars.sh
   │       ├── stderr.txt
   │       ├── stdout.txt
   │       └── test-run.toml
   ├── nccl.all_reduce
   │   └── 0
   │       ├── env_vars.sh
   │       ├── stderr.txt
   │       ├── stdout.txt
   │       └── test-run.toml
   └── slurm-job.toml

Most of the files are the same: output files, env vars script, test run metadata. The difference for single sbatch mode is that ``slurm-job.toml`` is created for the entire job as well as ``cloudai_sbatch_script.sh``. Extra ``common.err``/``common.out`` files are created for general sbatch outputs.

Extra srun and sbatch Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI forms sbatch script and srun commands following internal rules. Users can affect the generation by setting special arguments in the system TOML file.

For example (in a system TOML file):

.. code-block:: toml

   extra_sbatch_args = [
     "--section=4",
     "--other-arg val"
   ]

It will result in sbatch file content like this:

.. code-block:: bash

   ... # CloudAI set sbatch arguments
   #SBATCH --section=4
   #SBATCH --other-arg val
   ...

Another example (in a system TOML file):

.. code-block:: toml

   extra_srun_args = "--arg=val --other-arg=other-val"

Will result in srun command inside sbatch script like this:

.. code-block:: bash

   srun ... --arg=val --other-arg=other-val ...

Container Mounts
~~~~~~~~~~~~~~~~

Container-based Slurm workloads use mounts to expose data and results. To simplify file system related tasks, CloudAI mounts the following directories into the container:

1. Test output directory (``<output_path>/<scenario_name_with_timestamp>/<test_name>/<iteration>``, like ``results/scenario_2024-06-18_17-40-13/Tests.1/0``) is mounted as ``/cloudai_run_results``.
2. Test specific mounts can be specified in TOML files:

   .. code-block:: toml

      extra_container_mounts = [
        "/path/to/mount1:/path/in/container1",
        "/path/to/mount2:/path/in/container2"
      ]

      [cmd_args]
      ...

   These mounts are not verified for validity and do not override default mounts.

3. Test specific mounts can be mounted in-code.

Head Node without Shared Storage Available on Compute Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When compute nodes do not share the file system with head node, ``--enable-cache-without-check`` for ``run`` and ``dry-run`` skips the real check for cache existence, but still builds all paths correctly. The flow is like this:

1. *[on the head node]* run ``cloudai install``.
2. *[on the head node]* copy cache to compute nodes.
3. Modify ``system.toml`` to set compute nodes' installation root.
4. Run ``cloudai run --enable-cache-without-check ...``.

Dev Details
~~~~~~~~~~~

``SlurmCommandGenStrategy`` defines abstract method ``_container_mounts(tr: TestRun)`` that must be implemented by every subclass. This method is used in ``SlurmCommandGenStrategy.container_mounts(tr: TestRun)`` (defined as ``@final``) where mounts like ``/cloudai_run_results`` (default mount), ``TestDefinition.extra_container_mounts`` (from Test TOML) and test specific mounts (defined in-code) are added.

Nsys Tracing
~~~~~~~~~~~~

Users can enable Nsys tracing for any workload when running via Slurm. Note that ``nsys`` should be available on the compute nodes; CloudAI does not manage it.

The configuration fields are:

.. code-block:: python

   enable: bool = True
   nsys_binary: str = "nsys"
   task: str = "profile"
   output: Optional[str] = None
   sample: Optional[str] = None
   trace: Optional[str] = None
   force_overwrite: Optional[bool] = None
   capture_range: Optional[str] = None
   capture_range_end: Optional[str] = None
   cuda_graph_trace: Optional[str] = None
   gpu_metrics_devices: Optional[str] = None
   extra_args: list[str] = []

Fields with ``None`` value are not passed to ``nsys`` command.
