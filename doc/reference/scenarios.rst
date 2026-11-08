Scenario configuration
======================

A test scenario is a set of tests with specific dependencies between them. A test scenario is described in a TOML schema file. The following is an example of a test scenario file:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   num_nodes = "2"
   time_limit = "00:20:00"

   [[Tests]]
   id = "Tests.2"
   test_name = "nccl_test_all_gather"
   num_nodes = "2"
   time_limit = "00:20:00"
     [[Tests.dependencies]]
     type = "start_post_comp"
     id = "Tests.1"

   [[Tests]]
   id = "Tests.3"
   test_name = "nccl_test_reduce_scatter"
   num_nodes = "2"
   time_limit = "00:20:00"
     [[Tests.dependencies]]
     type = "start_post_comp"
     id = "Tests.2"

The scenario ``name`` identifies the experiment. Each ``[[Tests]]`` entry has a unique ``id``.
Use ``test_name`` to select a definition from the test directory, or place the workload definition directly in the entry.
See :doc:`/guides/configuration` for the smaller inline form.

There are two ways to specify nodes:

- Using the ``num_nodes`` field as shown in the example
- Specifying nodes explicitly like ``nodes = ["node-001", "node-002"]``

**Note:** When an explicit node list is provided (e.g., ``nodes = ["node-001", "node-002"]``), CloudAI lets Slurm apply the arbitrary distribution policy for task placement.

Alternatively, you can utilize the groups feature in the system schema to specify nodes like ``nodes = ['PARTITION_NAME:GROUP_NAME:NUM_NODES']``, which allocates ``num_nodes`` from the group name in the specified partition. You can also use ``nodes = ['PARTITION_NAME:GROUP_NAME:max_avail']``, which allocates all the available nodes from the group name in the specified partition.

You can optionally specify a time limit in the Slurm format. Tests can have dependencies. If no dependencies are specified, all tests will run in parallel.

CloudAI supports three types of dependencies:

- ``start_post_init``
- ``start_post_comp``
- ``end_post_comp``

Dependencies of a test can be described as a subsection of the test. It requires other tests' ``id`` and dependency ``type``.

- ``start_post_init`` means the test starts after the prior test begins, with a specified delay.
- ``start_post_comp`` means the test starts after the prior test completes.
- ``end_post_comp`` means the test ends when the prior test completes.
