import logging
from cloudai.systems import LSFSystem
from cloudai.installer.slurm_installer import SlurmInstaller


class LSFInstaller(SlurmInstaller):
    """
    Installer for systems that use the LSF scheduler.

    Extends the SlurmInstaller and customizes it for LSF-specific requirements.
    """

    PREREQUISITES = ("git", "bsub", "bhosts", "bjobs", "bkill")

    def __init__(self, system: LSFSystem):
        """
        Initialize the LSFInstaller with a system object.

        Args:
            system (LSFSystem): The system schema object.
        """
        super().__init__(system)
        self.system = system

    def _check_prerequisites(self):
        """
        Check for the presence of required binaries specific to LSF.

        Returns:
            InstallStatusResult: Result containing the status and any error message.
        """
        try:
            self._check_required_binaries()
            return super()._check_prerequisites()
        except EnvironmentError as e:
            return InstallStatusResult(False, str(e))

    def _check_required_binaries(self):
        """
        Check for the presence of required binaries specific to LSF.

        Raises:
            EnvironmentError: If any required binary is missing.
        """
        for binary in self.PREREQUISITES:
            if not self._is_binary_installed(binary):
                raise EnvironmentError(f"Required binary '{binary}' is not installed.")
