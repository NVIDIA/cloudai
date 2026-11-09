Bash Command
============

Run an existing shell command as a CloudAI case.

Check :doc:`compatibility </reference/workload-support>` and prepare :doc:`your execution environment </guides/environment>`.
Use the example below as the starting point; supply its workload-specific inputs before execution.

Inspect execution status and measurements using :doc:`/reporting`.

Adapt the workload
------------------

Bash Command workload (`test_template_name` is ``BashCmd``) allows users to execute arbitrary bash commands within the CloudAI framework. This is useful for simple scripts, custom testing commands, or integrating external tools.

``cmd`` specified in the ``cmd_args`` section will be added as-is into generated sbatch script.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_bash_test"
   description = "Example bash command test"
   test_template_name = "BashCmd"

   [cmd_args]
   cmd = "echo 'Hello from CloudAI!'"

Test Scenario example:

.. code-block:: toml

   name = "bash-test"

   [[Tests]]
   id = "bash.1"
   num_nodes = 1
   time_limit = "00:05:00"

   test_name = "my_bash_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "bash-test"

   [[Tests]]
   id = "bash.1"
   num_nodes = 1
   time_limit = "00:05:00"

   name = "my_bash_test"
   description = "Example bash command test"
   test_template_name = "BashCmd"

     [Tests.cmd_args]
     cmd = "echo 'Hello from CloudAI!'"

Configuration and API reference
-------------------------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.bash_cmd.bash_cmd.BashCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.bash_cmd.bash_cmd.BashCmdTestDefinition
   :members:
   :show-inheritance:
