Prepare and clean up with hooks
===============================

A test hook in CloudAI is a specialized test that runs either before or after each main test in a scenario, providing flexibility to prepare the environment or clean up resources. Hooks are defined as pre-test or post-test and referenced in the test scenario’s TOML file using ``pre_test`` and ``post_test`` fields.

.. code-block:: toml

   name = "nccl-test"

   pre_test = "nccl_test_pre"
   post_test = "nccl_test_post"

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   time_limit = "00:20:00"

CloudAI organizes hooks in a dedicated directory structure:

- Hook directory: All hook configurations reside in ``conf/hook/``
- Hook test scenarios: Place pre-test hook and post-test hook scenario files in ``conf/hook/``
- Hook tests: Place individual tests referenced in hooks within ``conf/hook/test/``

In the execution flow, pre-test hooks run before the main test, which only executes if the pre-test completes successfully. Post-test hooks follow the main test, provided the prior steps succeed. If a pre-test hook fails, the main test and its post-test hook are skipped.

.. code-block:: toml

   name = "nccl_test_pre"

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   time_limit = "00:20:00"
