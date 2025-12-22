Workloads Requirements Installation
===================================

CloudAI workloads can define multiple "installables" as prerequisites. It can be a container image, git repository, HF model, etc.


Set Up Access to the Private NGC Registry
-----------------------------------------

First, make sure you have access to the Docker repository. Follow the following steps:

1. **Sign In**: Go to `NGC signin`_ and sign in with your credentials.
2. **Generate API Key**:
    - On the top right corner, click on the dropdown menu next to your profile
    - Select "Setup"
    - In the "Setup" section, find "Keys/Secrets"
    - Click "Generate API Key" and confirm when prompted. A new API key will be presented
    - **Note**: Save this API key locally as you will not be able to view it again on NGC

.. _NGC signin: https://ngc.nvidia.com/signin

Next, set up your enroot credentials. Ensure you have the correct credentials under ``~/.config/enroot/.credentials``:

.. code-block:: text

    machine nvcr.io login $oauthtoken password <api-key>

Replace `<api-key>` with your respective credentials. Keep `$oauthtoken` as is.


🤗 Hugging Face Models
----------------------

Some workloads require Hugging Face models. CloudAI will download the models from Hugging Face and cache them in the location specified by System's ``hf_home_path`` field. By default, it is set to ``<INSTALL_DIR>/huggingface``, but any other location can be specified. When Slurm is used, this location will be mounted to the container.


Authentication with Hugging Face
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of now, CloudAI doesn't handle authentication with Hugging Face, so it is up to the user to enable authentication with Hugging Face in the shell where CloudAI is run. Users might need to run the following command:

.. code-block:: bash

    uv run hf auth login

Once done, all Hugging Face models will be downloaded using existing authentication.
