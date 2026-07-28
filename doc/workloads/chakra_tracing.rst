Chakra / Kineto Tracing with MegatronBridge
============================================

Disabling nsys
--------------

The existing MegatronBridge TOMLs in ``conf/staging/`` have ``enable_nsys = true``.
To switch to Kineto/Chakra profiling, either:

- **Omit** ``enable_nsys`` from the TOML (it defaults to ``false``)
- **Or** explicitly set ``enable_nsys = false``

Both are equivalent. Do not set ``enable_nsys = true`` alongside Kineto flags — the
two profilers conflict and Megatron-Bridge will raise an assertion error.


Enabling Kineto Trace
---------------------

Add the following to ``[extra_cmd_args]`` in your Test TOML. These are Hydra-style
overrides passed directly to the Megatron-Bridge training script:

.. code-block:: toml

   [extra_cmd_args]
   "profiling.use_pytorch_profiler" = "true"
   "profiling.profile_step_start" = "45"
   "profiling.profile_step_end" = "50"
   "profiling.profile_ranks" = "[0]"
   "profiling.record_shapes" = "true"
   "profiling.record_memory_history" = "true"
   "profiling.memory_snapshot_path" = "memory_profile.pickle"

This produces the following under ``<cloudai_output>/``:

- ``torch_profile/rank-0.json.gz`` — Chrome-format kineto trace (always produced)
- ``memory_profile.pickle`` — memory allocation snapshot (in nested experiment dir)


Enabling Chakra Trace
---------------------

The Chakra trace is a PyTorch ``ExecutionTraceObserver`` recording — a separate file
from the kineto trace. Add ``pytorch_profiler_collect_chakra = true``:

.. code-block:: toml

   [extra_cmd_args]
   "profiling.use_pytorch_profiler" = "true"
   "profiling.profile_step_start" = "45"
   "profiling.profile_step_end" = "50"
   "profiling.profile_ranks" = "[0]"
   "profiling.record_shapes" = "true"
   "profiling.record_memory_history" = "true"
   "profiling.memory_snapshot_path" = "memory_profile.pickle"
   "profiling.pytorch_profiler_collect_chakra" = "true"

This produces an additional ``chakra/rank-0.json.gz`` under ``<cloudai_output>/``.
The kineto trace (``torch_profile/rank-0.json.gz``) is always produced regardless of
this flag.


Reference TOML
--------------

A ready-to-use TOML is provided at:

.. code-block:: text

   conf/experimental/megatron_bridge/test/gb200/megatron_bridge_nemotron_nano_pytorch_profiler.toml

This runs Nemotron 3 Nano 3B (2 nodes, 8 GPUs, 60 steps) with Kineto profiling at
steps 45–50 and is suitable for quickly validating the profiling setup.


Parameter Reference
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Parameter
     - Type
     - Description
   * - ``profiling.use_pytorch_profiler``
     - bool
     - Enable PyTorch kineto profiler. Required for all Kineto/Chakra tracing.
   * - ``profiling.profile_step_start``
     - int
     - Training step at which profiling begins.
   * - ``profiling.profile_step_end``
     - int
     - Training step at which profiling ends and trace is exported.
   * - ``profiling.profile_ranks``
     - list[int]
     - GPU ranks to profile (e.g. ``[0]`` for rank 0 only).
   * - ``profiling.record_shapes``
     - bool
     - Record tensor shapes in the kineto trace.
   * - ``profiling.record_memory_history``
     - bool
     - Record memory allocation history (produces ``memory_profile.pickle``).
   * - ``profiling.memory_snapshot_path``
     - str
     - Filename for the memory snapshot pickle file.
   * - ``profiling.pytorch_profiler_collect_chakra``
     - bool
     - Collect Chakra ExecutionTraceObserver trace (``chakra/rank-N.json.gz``). The kineto trace is always produced; this adds an additional file.
   * - ``profiling.pytorch_profiler_collect_callstack``
     - bool
     - Include Python call stack in the kineto trace (increases trace size).
   * - ``enable_nsys``
     - bool
     - Enable Nsight Systems profiling. **Cannot be used with Kineto.** Default: ``false``.
