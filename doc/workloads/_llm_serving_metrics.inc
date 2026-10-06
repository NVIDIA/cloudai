Structured metrics
~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: "Metric", "Unit", "Preferred", "Dimensions"

   "``request_throughput``", "requests/s", "Higher", "None"
   "``output_token_throughput``", "tokens/s", "Higher", "None"
   "``ttft`` (time to first token)", "ms", "Lower", "``statistic``: mean, median, p99"
   "``tpot`` (time per output token)", "ms", "Lower", "``statistic``: mean, median, p99"
   "``accuracy``", "ratio (0–1)", "Higher", "None"

Metric and dimensions identify each value within an iteration or DSE step. Model, backend, and configured concurrency
remain in the test/run context. Only available, finite measurements are emitted; accuracy requires
``semantic_eval_cmd_args``.
