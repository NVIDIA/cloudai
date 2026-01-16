Chakra Replay
=============

This workload (`test_template_name` is ``ChakraReplay``) replays execution traces from the Chakra execution trace format for performance analysis and debugging.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_chakra_test"
   description = "Example Chakra replay test"
   test_template_name = "ChakraReplay"

   [cmd_args]
   trace_path = "/path/to/trace.et"

Test Scenario example:

.. code-block:: toml

   name = "chakra-replay-test"

   [[Tests]]
   id = "chakra.1"
   num_nodes = 1
   time_limit = "00:10:00"

   test_name = "my_chakra_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "chakra-replay-test"

   [[Tests]]
   id = "chakra.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "my_chakra_test"
   description = "Example Chakra replay test"
   test_template_name = "ChakraReplay"

     [Tests.cmd_args]
     trace_path = "/path/to/trace.et"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.chakra_replay.chakra_replay.ChakraReplayCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.chakra_replay.chakra_replay.ChakraReplayTestDefinition
   :members:
   :show-inheritance:
