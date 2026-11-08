Publish to an HTTP repository
=============================

The HTTP Data Repository is currently supported for Slurm systems only. To enable access, you must update your system schema file and create a credential file in your CloudAI project's root directory.

The following steps are required to configure the HTTP Data Repository:

- :ref:`Step 1: Update the System Schema File <step-1-update-system-schema-file>`
- :ref:`Step 2: Create the Credential File <step-2-create-credential-file>`
- :ref:`Step 3: Usage <step-3-usage>`

.. _step-1-update-system-schema-file:

Updating the System Schema File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the following section to your system schema TOML file (e.g., ``system_schema.toml``):

.. code-block:: toml

   [data_repository]
   endpoint = "https://my-data-endpoint.com"

Replace the endpoint with your actual data repository URL.

.. _step-2-create-credential-file:

Creating the Credential File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the root of your CloudAI project (i.e., the current working directory), create a file named ``.cloudai.toml`` with the following content:

.. code-block:: toml

   [data_repository]
   token = "<your-api-token-here>"

Replace ``<your-api-token-here>`` with your actual token.

.. _step-3-usage:

Usage
~~~~~

Both the endpoint and token must be valid for the HTTP Data Repository to function correctly. If either is missing or incorrect, data will not be posted.
