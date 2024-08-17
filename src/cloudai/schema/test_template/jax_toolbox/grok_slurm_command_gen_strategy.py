class GrokSlurmCommandGenStrategy(JaxToolboxSlurmCommandGenStrategy):
    """Command generation strategy for Grok tests on Slurm systems."""

    def _handle_threshold_and_env(
        self, cmd_args: Dict[str, str], env_vars: Dict[str, str], combine_threshold_bytes: int, num_nodes: int
    ):
        key = "Grok.XLA_FLAGS.combine_threshold_bytes"
        del cmd_args[key]

        setup_flags_key = "Grok.setup_flags.gpus_per_node"
        per_gpu_combine_threshold = int(combine_threshold_bytes / (int(cmd_args[setup_flags_key]) * num_nodes))
        env_vars["PER_GPU_COMBINE_THRESHOLD"] = str(per_gpu_combine_threshold)
