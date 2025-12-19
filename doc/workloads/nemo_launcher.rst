NeMo v1.0 aka NemoLauncher (DEPRECATED)
=======================================
This workload is **DEPRECATED**. Please use :doc:`nemo_run` instead.


Downloading and Installing the NeMo Dataset (The Pile Dataset)
--------------------------------------------------------------
This section describes how users can download the NeMo datasets on the server. The install mode of CloudAI handles the installation of all test prerequisites, but downloading and installing datasets is not the responsibility of the install mode. This is because any large datasets should be installed globally by the administrator and shared with multiple users, even if a user does not use CloudAI.

For CloudAI users, we provide a detailed guide about downloading and installing the NeMo datasets in this section. By default, the NeMo launcher uses mock datasets for testing purposes. If you want to run tests using real datasets, you must download the datasets and update the test `.toml` files accordingly to locate the datasets and provide appropriate prefixes.

To understand the datasets available in the NeMo framework, you can refer to the Data Preparation section of `the document <https://docs.nvidia.com/launchpad/ai/base-command-nemo/latest/bc-nemo-step-02.html#use-bignlp-to-download-and-prepare-the-pile-dataset>`_. According to the document, you can download and use the Pile dataset. The document also provides detailed instructions on how to download these datasets for various platforms.

  Let’s assume that we have a Slurm cluster.

You can download the datasets with the following command:

.. code-block:: bash

   $ git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git
   $ cd NeMo-Framework-Launcher
   $ python3 launcher_scripts/main.py \
       container=nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11\
       stages=["data_preparation"]\
       launcher_scripts_path=$PWD/launcher_scripts\
       base_results_dir=$PWD/result\
       env_vars.TRANSFORMERS_OFFLINE=0\
       data_dir=directory_path_to_download_dataset\
       data_preparation.run.time_limit="96:00:00"

Once you submit a NeMo job with the data preparation stage, you should be able to find data downloading jobs with the ``squeue`` command. If this command does not work, please review the log files under ``$PWD/result``. If you want to download the full Pile dataset, you should have at least 1TB of space in the directory to download the dataset because the Pile dataset size is 800GB.
By default, NeMo will look at the configuration file under ``conf/config.yaml``:

.. code-block:: yaml

   defaults:
     - data_preparation: baichuan2/download_baichuan2_pile

   stages:
     - data_preparation

As the data preparation field points to ``baichuan2/download_baichuan2_pile``, it will read the YAML file:

.. code-block:: yaml

   run:
     name: download_baichuan2_pile
     results_dir: ${base_results_dir}/${.name}
     time_limit: "4:00:00"
     dependency: "singleton"
     node_array_size: 30
     array: ${..file_numbers}
     bcp_preproc_npernode: 2 # 2 should be safe to use and x2 times faster.

   dataset: pile
   download_the_pile: True  # Whether to download the pile dataset from the internet.
   the_pile_url: "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/"  # Source URL to download The Pile dataset from.
   file_numbers: "0-29"  # The pile dataset consists of 30 files (0-29), choose which ones to download.
   preprocess_data: True  # True to preprocess the data from a jsonl file, False otherwise.
   download_tokenizer_url: "https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/tokenizer.model"
   tokenizer_typzer_library: "sentencepiece"
   tokenizer_save_dir: ${data_dir}/baichuan2
   tokenizer_model:  ${.tokenizer_save_dir}/baichuan2_tokenizer.model
   rm_downloaded: False # Extract script will remove downloaded zst after extraction
   rm_extracted: False # Preprocess script will remove extracted files after preproc.

You can update the fields to adjust the behavior. For example, you can update the ``file_numbers`` field to adjust the number of dataset files to download. This will allow you to save disk space.

Note: For running Nemo Llama model
------------------------------------

It is important to follow these additional steps:

1. Go to `🤗 Hugging Face <https://huggingface.co/docs/transformers/en/model_doc/llama>`_.
2. Follow the instructions on how to download the tokenizer.
3. Replace ``TOKENIZER_MODEL`` in ``training.model.tokenizer.model=TOKENIZER_MODEL`` with your path (the tokenizer should be a ``.model`` file) in ``conf/common/test/llama.toml``.


Troubleshooting
---------------

* If your run is not successful, please review the stderr and stdout files generated under the results directory. Within the output directory, locate the run directory, and under the run directory, you will find stderr files like ``log-nemo-megatron-run_[job_id].err``. Please review these files for any meaningful error messages
* Trying the CloudAI-generated NeMo launcher command can be helpful as well. You can find the executed command in your stdout and in your log file (debug.log) in your current working directory. Review and run the command, and you can modify the arguments to troubleshoot the issue