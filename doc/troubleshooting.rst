Troubleshooting
===============

This section guides you through identifying the root cause of issues, determining whether they stem from system infrastructure or a bug in CloudAI.

Identifying the Root Cause
---------------------------

If you encounter issues running a command, start by reading the error message to understand the root cause. We strive to make our error messages and exception messages as readable and interpretable as possible.

System Infrastructure vs. CloudAI Bugs
---------------------------------------

To determine whether an issue is due to system infrastructure or a CloudAI bug, follow these steps:

1. **Check stdout Messages:** If CloudAI fails to run a test successfully, it will be indicated in the stdout messages that a test has failed.

2. **Review Log Files:**

   - Navigate to the output directory and review ``debug.log``, stdout, and stderr files
   - ``debug.log`` contains detailed steps executed by CloudAI, including generated commands, executed commands, and error messages

3. **Analyze Error Messages:** By examining the error messages in the log files, you can understand the type of errors CloudAI encountered.

4. **Examine Output Directory:** If a test fails without explicit error messages, review the output directory of the failed test. Look for ``stdout.txt``, ``stderr.txt``, or any generated files to understand the failure reason.

5. **Manual Rerun of Tests:**

   - To manually rerun the test, consult the ``debug.log`` for the command CloudAI executed
   - Look for an ``sbatch`` command with a generated ``sbatch`` script
   - Execute the command manually to debug further

If the problem persists, please report the issue at https://github.com/NVIDIA/cloudai/issues/new/choose. When you report an issue, please make sure it is reproducible. Follow the issue template and provide any necessary details, such as the hash commit used, system settings, any changes in the schema files, and the command.
