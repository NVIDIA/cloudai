AIConfigurator
==============

This workload (`test_template_name` is ``Aiconfigurator``) runs the AIConfigurator predictor using the installed
``aiconfigurator`` Python package. It is a **Standalone** workload (no Slurm/Kubernetes/RunAI required).

Outputs
-------

Each test run produces:

- ``report.json``: Predictor output (JSON dict of metrics and metadata)
- ``stdout.txt`` / ``stderr.txt``: Predictor logs
- ``run_simple_predictor.sh``: Repro script containing the exact executed command (useful for debugging)

Usage Example
-------------

Test TOML example (Disaggregated mode):

.. code-block:: toml

   name = "aiconfigurator_disagg_demo"
   description = "Example AIConfigurator disaggregated predictor"
   test_template_name = "Aiconfigurator"

   [cmd_args]
   model_name = "LLAMA3.1_70B"
   system = "h200_sxm"
   backend = "trtllm"
   version = "0.20.0"
   isl = 4000
   osl = 500

     [cmd_args.disagg]
     p_tp = 1
     p_pp = 1
     p_dp = 1
     p_bs = 1
     p_workers = 1

     d_tp = 1
     d_pp = 1
     d_dp = 1
     d_bs = 8
     d_workers = 2

     prefill_correction_scale = 1.0
     decode_correction_scale = 1.0

Test TOML example (Aggregated/IFB mode):

.. code-block:: toml

   name = "aiconfigurator_agg_demo"
   description = "Example AIConfigurator aggregated predictor"
   test_template_name = "Aiconfigurator"

   [cmd_args]
   model_name = "LLAMA3.1_70B"
   system = "h200_sxm"
   backend = "trtllm"
   version = "0.20.0"
   isl = 4000
   osl = 500

     [cmd_args.agg]
     batch_size = 8
     ctx_tokens = 16
     tp = 1
     pp = 1
     dp = 1

Running
-------

.. code-block:: bash

   uv run cloudai run --system-config conf/common/system/standalone_system.toml \
      --tests-dir conf/experimental/aiconfigurator/test \
      --test-scenario conf/experimental/aiconfigurator/test_scenario/aiconfigurator_disagg.toml

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.aiconfig.aiconfigurator.AiconfiguratorCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.aiconfig.aiconfigurator.AiconfiguratorTestDefinition
   :members:
   :show-inheritance:

Command Generation Strategy (Standalone)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.aiconfig.standalone_command_gen_strategy.AiconfiguratorStandaloneCommandGenStrategy
   :members:
   :show-inheritance:

Report Generation Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.aiconfig.report_generation_strategy.AiconfiguratorReportGenerationStrategy
   :members:
   :show-inheritance:


