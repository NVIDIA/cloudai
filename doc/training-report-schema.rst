Training Report Schema
======================

Use this schema to interpret training report fields. Preserve metric units and aggregation windows when comparing experiments.

``training_report.json`` is the unified training output for NeMoRun, MegatronRun, and Megatron-Bridge.

Versioning
----------

The top-level ``schema_version`` field uses ``MAJOR.MINOR`` versioning:

- Increment **MAJOR** when removing, renaming, or incompatibly changing a field.
- Increment **MINOR** when adding an optional field or otherwise making a backward-compatible schema change.
- Do not change the schema version for implementation fixes that leave the JSON contract unchanged.

Consumers should reject unsupported major versions and tolerate unknown fields within a supported major version.

Version History
---------------

1.0 — 2026-07-14
~~~~~~~~~~~~~~~~

Added to ``root``:

- ``schema_version``: ``str``

Added to ``root.config``:

- ``test_id``: ``str``
- ``test_name``: ``str``
- ``description``: ``str``
- ``test_scenario_name``: ``str``
- ``system_path``: ``str``
- ``tests_dir_path``: ``str``
- ``test_scenario_path``: ``str``
- ``container_image``: ``str``
- ``cloudai_execution_node``: ``str``
- ``env_vars``: ``dict[str, Any]``
- ``gpus_per_node``: ``Optional[int]``
- ``nodes``: ``list[str]``
- ``clique_size``: ``Optional[int]``
- ``fp8``: ``Optional[str]``
- ``fp8_recipe``: ``Optional[str]``
- ``expert_tensor_parallel_size``: ``int``

Training Report Models
----------------------

.. autoclass:: cloudai.report_generator.training.models.TrainingResults
   :members:
   :exclude-members: __init__

.. autoclass:: cloudai.report_generator.training.models.TrainingConfig
   :members:
   :exclude-members: __init__

.. autoclass:: cloudai.report_generator.training.models.TrainingStep
   :members:
   :exclude-members: __init__

.. autoclass:: cloudai.report_generator.training.models.StepAggregation
   :members:
   :exclude-members: __init__, from_steps

.. autoclass:: cloudai.report_generator.training.models.MetricStats
   :members:
   :exclude-members: __init__, from_values
