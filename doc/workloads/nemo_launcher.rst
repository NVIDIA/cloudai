NeMo Launcher (deprecated)
==========================

This legacy integration is deprecated. Use :doc:`nemo_run` for NeMo training and fine-tuning,
or :doc:`megatron_bridge` for a maintained training starter.

Migrate an experiment
---------------------

Choose the matching model and training recipe in the replacement workload. Transfer dataset and tokenizer paths,
parallelism, and batch settings deliberately; the launcher's configuration is not interchangeable with the new recipe.
Run a short case and compare its output before scaling up.

Dataset preparation
-------------------

Provision the selected recipe's data on storage accessible to the execution nodes. Use that recipe's documented
format and tokenizer, then configure paths and mounts. See :doc:`/guides/model-access` and :doc:`/guides/environment`.
