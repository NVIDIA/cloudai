Chakra / Kineto Tracing with MegatronBridge
============================================

MegatronBridge supports two profiling modes that **cannot run simultaneously**:

- **nsys (Nsight Systems)** — GPU timeline profiling via ``enable_nsys = true``
- **Kineto / Chakra** — PyTorch kineto trace + optional Chakra trace via ``[extra_cmd_args]``

This page explains how to enable Kineto/Chakra tracing and disable nsys.


Background
----------

Kineto trace (``rank-0.json.gz``) is a Chrome-format trace viewable in tools like
`Perfetto <https://perfetto.dev/>`_ or ``chrome://tracing``.

Chakra trace is an additional execution graph format generated *on top of* the Kineto
trace. To produce a Chakra trace you must first enable the Kineto trace, then set
``pytorch_profiler_collect_chakra = true``.


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

This produces ``rank-0.json.gz`` (kineto trace) and ``memory_profile.pickle``
(memory snapshot) under ``<cloudai_output>/torch_profile/``.

.. note::

   CloudAI automatically copies the trace from NemoRun's nested experiment directory
   to ``<cloudai_output>/torch_profile/`` after the job completes, making the location
   consistent with MegatronRun.


Enabling Chakra Trace
---------------------

Add ``pytorch_profiler_collect_chakra = true`` alongside the Kineto flags:

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

Setting ``pytorch_profiler_collect_chakra = false`` disables the Chakra trace while
keeping the Kineto trace (``rank-0.json.gz``) and memory snapshot.


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
     - Collect Chakra execution graph trace (requires Kineto enabled).
   * - ``profiling.pytorch_profiler_collect_callstack``
     - bool
     - Include Python call stack in the kineto trace (increases trace size).
   * - ``enable_nsys``
     - bool
     - Enable Nsight Systems profiling. **Cannot be used with Kineto.** Default: ``false``.
