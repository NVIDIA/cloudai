Installation Requirements
==========================

CloudAI workloads can define multiple installables as prerequisites. The installable can be a container image, git repository, HF model, etc.


Python Executables from Git Repositories
----------------------------------------

Some workloads wrap a git repository in a ``PythonExecutable`` and install the
repository in a dedicated virtual environment. Such an environment can use a
different Python interpreter from the one running CloudAI. Set ``python_version``
on the repository to select that interpreter explicitly.

In a test definition:

.. code-block:: toml

    [[git_repos]]
    url = "https://github.com/NVIDIA-NeMo/Run.git"
    commit = "v0.10.0"
    python_version = "3.11.9"

In a test embedded in a scenario:

.. code-block:: toml

    [[Tests.git_repos]]
    url = "https://github.com/NVIDIA-NeMo/Run.git"
    commit = "v0.10.0"
    python_version = "3.11.9"

CloudAI selects the interpreter for a ``PythonExecutable`` in this order:

1. The repository's explicit ``python_version`` value.
2. The nearest ``.python-version`` file, searching from the executable's
   project subdirectory towards the repository root. The search never leaves
   the repository.
3. The interpreter running CloudAI (``sys.executable``), which preserves the
   behavior of repositories without a Python version setting.

Other version declarations, including ``.python-versions``, ``.tool-versions``,
``runtime.txt``, global uv configuration, and ``requires-python`` in
``pyproject.toml``, are not used for this selection.

CloudAI uses the uv executable bundled with its Python package for both
``PythonExecutable`` and ``PythonEnvironment`` installables. Neither installable
requires a separately installed ``uv`` command or ``uv`` on ``PATH``. If the
selected interpreter is unavailable locally, the bundled uv can download it
during the first installation, so that installation requires network access and
can take longer than subsequent runs. See `uv Python version management`_ for
details.

The ``python_version`` field does not by itself make a generic ``GitRepo``
executable. Repositories used only as mounts are still cloned and mounted; the
field is consumed only by workloads that wrap the repository in a
``PythonExecutable``.

After upgrading CloudAI, an existing virtual environment that uses a repository
``.python-version`` pin might be recreated once. CloudAI records the effective
interpreter request for future checks and rebuilds a pinned legacy environment
when that record is missing or no longer matches.

.. _uv Python version management: https://docs.astral.sh/uv/concepts/python-versions/


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

Authentication with Hugging Face
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of now, CloudAI does not handle authentication with Hugging Face, so it is up to the user to enable authentication with Hugging Face in the shell where CloudAI is run. Users might need to run the following command:

.. code-block:: bash

    uv run hf auth login

Once done, all Hugging Face models will be downloaded using existing authentication.
