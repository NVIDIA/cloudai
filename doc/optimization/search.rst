Run a configuration search
==========================

Compare two NCCL algorithm settings and inspect the configuration with the best observed reward.

Prepare a baseline
------------------

Run the :ref:`networking example <networking-with-nccl>` first. This search uses the same two-node Slurm setup and public image.
Keep the system, message sizes, and other workload settings fixed.

Run two candidates
------------------

.. code-block:: bash

   uv run cloudai run --system-config slurm.toml \
       --test-scenario conf/experimental/quickstart/nccl-search.toml

.. literalinclude:: ../../conf/experimental/quickstart/nccl-search.toml
   :language: toml
   :start-at: name =

``NCCL_ALGO`` defines two choices. Built-in ``grid_search`` evaluates both; it does not adapt its choices from
measurements. The ``default`` metric is mean in-place latency in microseconds across the reported message sizes; ``inverse`` converts a lower-is-better
metric into a higher-is-better reward. Check :doc:`objectives` before adapting the objective.

Inspect the selected result
---------------------------

Read ``tests[].dse`` in ``experiment.json`` for search details and the selected step; ``tests[].metrics`` contains
the successful step with the highest valid reward. Inspect the DSE report and selected test configuration alongside
the workload logs. See :doc:`/reporting` for a real two-candidate example and its measurement context.

Expand the search
-----------------

Add dimensions only after estimating their combined cost with :doc:`budget`. Keep the baseline and search in separate
scenarios. To choose trials using prior feedback, integrate :doc:`a custom agent </extend/agent>`.
