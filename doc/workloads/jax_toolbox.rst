JaxToolbox workloads (DEPRECATED)
=================================

This workload is **DEPRECATED**.

Troubleshooting Steps for Grok
------------------------------

If an error occurs, follow these steps sequentially:

1. **Read the Error Messages**:
    Begin by reading the error messages printed by CloudAI. We strive to make our error messages clear and informative, so they are a good starting point for troubleshooting

2. **Review profile_stderr.txt**: JaxToolbox operates in two stages: the profiling phase and the actual run phase. We follow the PGLE workflow as described in the `PGLE workflow documentation <https://github.com/google/paxml?tab=readme-ov-file#run-pgle-workflow-on-gpu>`_. All stderr and stdout messages from the profiling phase are stored in ``profile_stderr.txt``. If the profiling stage fails, you should find relevant error messages in this file. Attempt to understand the cause of the error from these messages.

3. **Check the Actual Run Phase**:
   If the profiling stage completes successfully, CloudAI moves on to the actual run phase. The actual run generates stdout and stderr messages in separate files for each rank. Review these files to diagnose any issues during this phase.

Common Errors
~~~~~~~~~~~~~

**DEADLINE_EXCEEDED**:
   When running JaxToolbox on multiple nodes, the nodes must be able to communicate to execute a training job collaboratively. The DEADLINE_EXCEEDED error indicates a failure in the connection during the initialization stage. Potential causes include:

   - Hostname resolution failure by the slave nodes
   - The port opened by the master node is not accessible by other nodes
   - Network interface malfunctions
   - Significant time gap in the initialization phase among nodes. If one node starts early while others are still loading the Docker image, this error can occur. This can happen when a Docker image is not locally cached, and all nodes try to download it from a remote registry without sufficient network bandwidth. The resulting difference in initialization times can lead to a timeout on some nodes
