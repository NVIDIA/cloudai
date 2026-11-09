DDLB
====

Run distributed training experiments with DDLB.

Check :doc:`compatibility </reference/workload-support>` and prepare :doc:`your execution environment </guides/environment>`.
Use the example below as the starting point; supply its workload-specific inputs before execution.

Inspect execution status and measurements using :doc:`/reporting`.

Adapt the workload
------------------

DDLB workload (`test_template_name` is ``DDLB``) allows users to execute DDLB (Distributed Deep Learning Benchmarks) within the CloudAI framework. Please find the DDLB README at https://github.com/samnordmann/ddlb.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_ddlb_test"
   description = "Example DDLB test"
   test_template_name = "DDLB"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   primitive = "tp_columnwise"
   dtype = "float16"

Test Scenario example:

.. code-block:: toml

   name = "ddlb-test"

   [[Tests]]
   id = "ddlb.1"
   num_nodes = 1
   time_limit = "00:10:00"

   test_name = "my_ddlb_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "ddlb-test"

   [[Tests]]
   id = "ddlb.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "my_ddlb_test"
   description = "Example DDLB test"
   test_template_name = "DDLB"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     primitive = "tp_columnwise"
     m = 1024
     n = 128
     k = 1024
     dtype = "float16"
     num_iterations = 50
     num_warmups = 5
     impl = "pytorch;backend=nccl;order=AG_before"

Configuration and API reference
-------------------------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.ddlb.ddlb.DDLBCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.ddlb.ddlb.DDLBTestDefinition
   :members:
   :show-inheritance:
