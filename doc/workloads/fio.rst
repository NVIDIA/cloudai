fio
===

This workload (``test_template_name`` is ``Fio``) runs the open-source ``fio`` tool from CloudAI.
It supports Slurm and standalone systems, fio job files, direct CLI options, hooks on Slurm, and
DSE sweeps through list-valued ``cmd_args.args`` entries such as ``bs`` or ``iodepth``.

Usage Example
-------------

CLI options:

.. code-block:: toml

    name = "fio_randwrite"
    description = "fio random write"
    test_template_name = "Fio"

    [cmd_args]
    fio_binary = "fio"
    num_tasks_per_node = 1

      [cmd_args.args]
      name = "randwrite"
      filename = "/tmp/cloudai-fio-test"
      rw = "randwrite"
      bs = ["128k", "1m"]
      size = "80m"
      iodepth = [1, 8, 32]
      numjobs = 1
      group_reporting = true
      thread = true

Job file:

.. code-block:: toml

    name = "fio_job_file"
    description = "fio from a job file"
    test_template_name = "Fio"

    [cmd_args]
    fio_binary = "/tmp/fio/fio"
    job_file = "/tmp/kv_emulation.fio"

Test Scenario example:

.. code-block:: toml

    name = "fio-scenario"

    [[Tests]]
    id = "Tests.fio"
    test_name = "fio_randwrite"
    num_nodes = 2
    time_limit = "00:10:00"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.fio.fio.FioCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.fio.fio.FioTestDefinition
   :members:
   :show-inheritance:
