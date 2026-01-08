UCC
===

This workload (`test_template_name` is ``UCCTest``) allows users to execute UCC benchmarks within the CloudAI framework.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

    name = "ucc"
    description = "Example UCC test"
    test_template_name = "UCCTest"

    [cmd_args]
    docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"

Test Scenario example:

.. code-block:: toml

    name = "ucc-test"

    [[Tests]]
    id = "ucc.1"
    num_nodes = 1
    time_limit = "00:02:00"

    test_name = "ucc"

Test-in-Scenario example:

.. code-block:: toml

    name = "ucc-test"

    [[Tests]]
    id = "ucc.1"
    num_nodes = 1
    time_limit = "00:02:00"

    name = "ucc"
    description = "Example UCC test"
    test_template_name = "UCCTest"

    [Tests.cmd_args]
    docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.ucc_test.ucc.UCCCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.ucc_test.ucc.UCCTestDefinition
   :members:
   :show-inheritance:
