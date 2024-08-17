class NemotronSlurmCommandGenStrategy(JaxToolboxSlurmCommandGenStrategy):
    """Command generation strategy for Nemotron tests on Slurm systems."""

    def _handle_threshold_and_env(
        self, cmd_args: Dict[str, str], env_vars: Dict[str, str], combine_threshold_bytes: int, num_nodes: int
    ):
        keys = [
            "Nemotron.XLA_FLAGS.xla_gpu_all_reduce_combine_threshold_bytes",
            "Nemotron.XLA_FLAGS.xla_gpu_all_gather_combine_threshold_bytes",
            "Nemotron.XLA_FLAGS.xla_gpu_reduce_scatter_combine_threshold_bytes",
        ]
        key = next((k for k in keys if k in cmd_args), None)
        if key is None:
            raise ValueError("None of the Nemotron specific keys are found in cmd_args.")

        del cmd_args[key]

        setup_flags_key = "Nemotron.setup_flags.gpus_per_node"
        per_gpu_combine_threshold = int(combine_threshold_bytes / (int(cmd_args[setup_flags_key]) * num_nodes))
        env_vars["PER_GPU_COMBINE_THRESHOLD"] = str(per_gpu_combine_threshold)
