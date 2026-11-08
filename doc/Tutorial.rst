Run your first experiment
=========================

Run a small file I/O experiment locally and find its results. This path needs no GPU or scheduler.

Prepare
--------

:doc:`Install CloudAI <getting-started>` and run the following commands from the repository root.
Install the Fio executable; on Ubuntu, use ``sudo apt-get install fio``.

.. code-block:: bash

   cp conf/common/system/standalone_system.toml standalone.toml

Run
--------

The example writes an 80 MiB file at ``/tmp/cloudai-fio-quickstart``. Choose another filename in the scenario
if you want to measure a different filesystem.

.. code-block:: bash

   uv run cloudai run --system-config standalone.toml \
       --test-scenario conf/experimental/quickstart/fio.toml

.. dropdown:: View the scenario

   .. literalinclude:: ../conf/experimental/quickstart/fio.toml
      :language: toml
      :start-at: name =

Inspect
--------

Open the results directory printed by CloudAI. Inspect ``experiment.json`` for execution status and recorded metrics,
then the workload output for bandwidth and IOPS. A successful process alone does not establish that you measured
the intended device: check the configured path, I/O pattern, block size, and file size.
See :doc:`reporting` for result interpretation.

Adapt
--------

Change one Fio setting at a time and compare results under the same conditions.
Use :doc:`guides/configuration` to understand the two configuration files.
For GPU or multi-node experiments, choose a workload from :doc:`workloads/index` and follow
:doc:`guides/environment`. For configuration search, continue to :doc:`optimization/search`.
