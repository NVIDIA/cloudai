Installation Requirements
==========================

CloudAI workloads can define multiple installables as prerequisites. The installable can be a container image, git repository, HF model, etc.


Setting Up Access to the Private NGC Registry
---------------------------------------------

First, make sure you have access to the Docker repository. Proceed as follows:

1. **Sign In**: Go to `NGC signin`_ and sign in with your credentials.
2. **Generate API Key**:
    - On the top right corner, click on the dropdown menu next to your profile
    - Select **Setup**
    - In the **Setup** section, find **Keys/Secrets**
    - Click **Generate API Key** and confirm when prompted. A new API key will be presented
    - **Note**: Save this API key locally as you will not be able to view it again on NGC
    - Set up your enroot credentials. Make sure you have the correct credentials under **~/.config/enroot/.credentials**:

        .. code-block:: text

            machine nvcr.io login $oauthtoken password <api-key>

    - Replace `<api-key>` with your respective credentials. Keep `$oauthtoken` as is.


.. _NGC signin: https://ngc.nvidia.com/signin


Hugging Face Models
-------------------

Some workloads require Hugging Face models. CloudAI will download the models from Hugging Face and cache them in the location specified by System's ``hf_home_path`` field. By default, it is set to ``<INSTALL_DIR>/huggingface``, but any other location can be specified. When Slurm is used, this location will be mounted to the container.

Slurm systems can optionally set ``hf_local_home_path`` to an absolute compute-node local path. CloudAI will continue
to install and validate models in the shared ``hf_home_path``, then automatically copy each model required by the job
to ``hf_local_home_path`` on every allocated node before starting any containers. The local cache is reused when the
same model revision is already present. If ``hf_local_home_path`` is not set, model installation and mounting behave
exactly as before.

Both paths must use the same absolute location on every allocated node. ``hf_home_path`` must be readable from the
compute nodes, while ``hf_local_home_path`` must be writable and should refer to local storage such as node-local NVMe.

Authentication with Hugging Face
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of now, CloudAI does not handle authentication with Hugging Face, so it is up to the user to enable authentication with Hugging Face in the shell where CloudAI is run. Users might need to run the following command:

.. code-block:: bash

    uv run hf auth login

Once done, all Hugging Face models will be downloaded using existing authentication.
