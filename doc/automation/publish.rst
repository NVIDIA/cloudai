Publish results and integrate with CI
=====================================

Collect a JUnit artifact for CI, or publish results to an object store.

Enable CI output
----------------

Add to the scenario:

.. code-block:: toml

   [reports]
   junit = { enable = true }

Collect ``junit.xml`` from the results directory in your CI system. It records regular iterations and search steps,
including failures. Preserve ``experiment.json`` and workload logs alongside it for diagnosis.

.. _uploading-results-to-object-storage:

Uploading Results to Object Storage
-----------------------------------

The ``s3`` scenario report publishes the scenario results directory to an
S3-compatible bucket. It is disabled by default, because shipping results off-box should
be a deliberate choice.

It is registered last, after ``tarball``, so it always observes the complete results
directory including every other report's output.

Files are uploaded under ``s3://<bucket>/<prefix>/<cluster>/<username>/<results-directory>/``,
where ``cluster`` is the system name and ``username`` is the original experiment owner's
``user`` field in ``experiment.json``, including when running ``generate-report``.
If the file is missing, unreadable, invalid, or has no recorded owner, uploads use
the current account's username. If that lookup also fails, uploads use ``unknown``.
Tarballs use the same path, with ``<results-directory>.tgz`` appended.

Install the optional dependency first:

.. code-block:: bash

   pip install 'cloudai[s3]'

Then enable it in a test scenario:

.. code-block:: toml

   [reports]
   s3 = { enable = true, bucket = "my-bucket", prefix = "cloudai/runs", upload_tarball = true }

Or, for Slurm systems, once per cluster in the system config:

.. code-block:: toml

   [reports]
   s3 = { enable = true, bucket = "my-bucket" }

Configuration options:

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``bucket``
     - ``$CLOUDAI_S3_BUCKET``
     - Destination bucket. Required; the upload is skipped with a warning if unset.
   * - ``prefix``
     - ``$CLOUDAI_S3_PREFIX``
     - Key prefix. Objects are written under ``<prefix>/<system_name>/<username>/<results_dir_name>/``.
   * - ``endpoint_url``
     - ``$CLOUDAI_S3_ENDPOINT_URL``
     - Custom endpoint, for MinIO or other S3-compatible stores.
   * - ``region``
     - unset
     - AWS region. When unset, boto3 resolves it (e.g. ``AWS_DEFAULT_REGION``).
   * - ``upload_tree``
     - ``true``
     - Upload each file individually, preserving relative paths.
   * - ``upload_tarball``
     - ``false``
     - Also upload a ``.tgz`` of the whole directory. It is created if absent and rebuilt if older than the results.
   * - ``upload_concurrency``
     - ``8``
     - Number of files uploaded concurrently when ``upload_tree`` is enabled. Must be at least 1.

At least one of ``upload_tree`` and ``upload_tarball`` must be enabled; the config is rejected otherwise.

Environment variables
~~~~~~~~~~~~~~~~~~~~~

The destination can be supplied through three environment variables, so a cluster-wide
default does not have to be repeated in every scenario:

.. list-table::
   :header-rows: 1

   * - Variable
     - Sets option
     - Notes
   * - ``CLOUDAI_S3_BUCKET``
     - ``bucket``
     - Destination bucket.
   * - ``CLOUDAI_S3_PREFIX``
     - ``prefix``
     - Key prefix. Empty by default.
   * - ``CLOUDAI_S3_ENDPOINT_URL``
     - ``endpoint_url``
     - Custom endpoint, for MinIO or other S3-compatible stores. An empty value is treated as unset.

A value set in TOML takes precedence over the environment variable. The variables are read by
the ``cloudai`` process when the report configuration is loaded, so export them where
``cloudai`` runs:

.. code-block:: bash

   export CLOUDAI_S3_BUCKET=my-bucket
   export CLOUDAI_S3_PREFIX=cloudai/runs
   export CLOUDAI_S3_ENDPOINT_URL=http://localhost:9000   # only for MinIO or other S3-compatible stores

With these set, enabling the report only needs ``s3 = { enable = true }``.

**Credentials are never read from CloudAI configuration.** They are resolved by boto3's
standard chain: ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``, ``~/.aws/credentials``,
or an instance/IAM role.

Because reports run inside a ``try``/``except``, an upload failure logs a warning and
leaves the run's exit status unchanged.

To upload a results directory from an earlier run, re-run the reports against it:

.. code-block:: bash

   cloudai generate-report --system-config <system.toml> --tests-dir <tests/> \
     --test-scenario <scenario.toml> --result-dir results/<scenario>_<timestamp>
