Install CloudAI
===============

CloudAI officially targets Python 3.14 while remaining compatible with Python 3.10 and newer.

Use uv 0.13.4 or newer to install and run CloudAI. For pip-based installation, use pip 26.2.2 or newer.
These versions include certificate trust updates for the eMudhra/emSign CA compromise.

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai
   uv run cloudai --help

Continue with :doc:`your first experiment <Tutorial>`. See :doc:`systems` and
:doc:`workloads/workloads_requirements_installation` for environment setup.

pip-based Installation
----------------------

See the preferred Python version in the ``.python-version`` file and make sure you have it installed
(For installation, see :ref:`install-custom-python-version`). Python 3.10 and newer remain supported.
Follow these steps:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai
   python3.14 -m venv venv
   source venv/bin/activate
   pip install -e .

.. note::

   When upgrading an existing installation, also update the host's ``ca-certificates`` package and the
   certificate bundles and installers in existing workload environments and container images.
   Updating CloudAI does not update these environments.


.. _install-custom-python-version:

Custom Python Version Installation
----------------------------------

If your system Python version is not supported, you can install a custom version using the `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ tool:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv venv --seed  # picks the python version from .python-version
                   # --seed installs pip and setuptools
   source .venv/bin/activate


See :doc:`reference/cli` for the command catalog.
