Plan compute usage
==================

Start with a small search space and estimate the cost before allocating resources.

Count candidates
----------------

For grid search, multiply the number of choices in each independent dimension.
Two algorithms and three message-size settings produce six candidates. Built-in grid search evaluates all of them;
``agent_steps`` does not cap that Cartesian product.

Estimate allocation cost
------------------------

.. code-block:: text

   GPU-hours = sum(allocated GPUs × allocated seconds for each evaluation) / 3600

Include setup, warmup, repetitions, and allocation overhead. Use a measured baseline to estimate duration.
Queue delay affects elapsed time separately. A per-job time limit is not an overall search-budget limit.

Control growth
--------------

Reduce dimensions and candidate counts first. Use short representative trials before expanding the space.
For feedback-driven selection or early stopping, use :doc:`/extend/agent` and verify the policy's budget behavior.
CloudAI's public agent interface permits custom policies; the framework does not imply a universal monetary or GPU-hour cap.
