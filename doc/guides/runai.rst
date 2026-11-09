Set up RunAI
============

The RunAI integration supports NCCL. For direct Kubernetes execution of Dynamo or NCCL, use :doc:`kubernetes`.

Configure your RunAI account and project in the system TOML:

.. code-block:: toml

   name = "runai-cluster"
   scheduler = "runai"

   install_path = "./install"
   output_path = "./results"

   base_url = "http://runai.example.com"       # The URL of your RunAI system, typically the same as used for the web interface.
   verify_certs = true                        # Applies to HTTPS requests and training log connections.
   user_email = "your_email"              # The email address used to log into the RunAI system.
   app_id = "your_app_id"                      # Obtained by creating an application in the RunAI web interface.
   app_secret = ""                             # Fill in the secret obtained with the app_id.
   project_id = "your_project_id"              # Project ID assigned or created in the RunAI system (usually an integer).
   cluster_id = "your_cluster_id"              # Cluster ID in UUID format (e.g., a69928cc-ccaa-48be-bda9-482440f4d855).

- After logging into the RunAI web interface, navigate to Access → Applications and create a new application to obtain ``app_id`` and ``app_secret``
- Use your assigned project and cluster IDs. Contact your administrator if they are not available
- All other fields follow the same semantics as in the Slurm system schema (e.g., ``install_path``, ``output_path``)

RunAI HTTP requests use a 30-second timeout. The optional ``verify_certs`` setting defaults to ``true`` and controls
server TLS certificate verification for both HTTPS requests and training log connections. Set it to ``false`` to disable
verification. For training log connections with a private certificate authority, install its CA certificate in the system trust store.
