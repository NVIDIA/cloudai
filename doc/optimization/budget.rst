Plan compute usage
==================

Start with a small search space and estimate the cost before allocating resources.

Count candidates
----------------

For grid search, multiply the number of choices in each independent dimension.
Two algorithms and three message-size settings produce six candidates.

For example, add this table to a case in an NCCL search scenario:

.. code-block:: toml

   [Tests.extra_env_vars]
   NCCL_ALGO = ["Ring", "Tree"]

This dimension contributes two candidates. Built-in grid search evaluates all of them;
``agent_steps`` does not cap that Cartesian product.

Estimate allocation cost
------------------------

.. code-block:: text

   GPU-hours = sum(allocated GPUs × allocated seconds for each evaluation) / 3600

* Estimate trial duration from a measured baseline.
* Include setup, warmup, repetitions, and allocation overhead.
* Track queue delay separately from allocated compute.
* Treat a per-job time limit separately from the overall search budget.

Constraint-rejected proposals are retried before consuming a DSE step, up to the configured consecutive-rejection
limit. See :ref:`constraint-rejected-proposals` for how skipped steps affect progress.

Control growth
--------------

Reduce dimensions and candidate counts first. Use short representative trials before expanding the space.
For feedback-driven selection or early stopping, use :doc:`/extend/agent` and verify the policy's budget behavior.
CloudAI's public agent interface permits custom policies; the framework does not imply a universal monetary or GPU-hour cap.
