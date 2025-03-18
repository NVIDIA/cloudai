from pydantic import BaseModel


class MetadataSystem(BaseModel):
    """Represents the system metadata."""

    user: str
    system_name: str
    os_type: str
    os_version: str
    linux_kernel_version: str
    gpu_arch_type: str
    cpu_model_name: str
    cpu_arch_type: str


class MetadataMPI(BaseModel):
    """Represents the MPI metadata."""

    mpi_type: str
    mpi_version: str
    hpcx_version: str


class MetadataCUDA(BaseModel):
    """Represents the CUDA metadata."""

    cuda_build_version: str
    cuda_runtime_version: str
    cuda_driver_version: str


class MetadataNetwork(BaseModel):
    """Represents the network metadata."""

    nics: str
    switch_type: str
    network_name: str
    mofed_version: str
    libfabric_version: str


class MetadataNCCL(BaseModel):
    """Represents the NCCL metadata."""

    version: str
    commit_sha: str


class SlurmSystemMetadata(BaseModel):
    """Represents the Slurm system metadata."""

    system: MetadataSystem
    mpi: MetadataMPI
    cuda: MetadataCUDA
    network: MetadataNetwork
    nccl: MetadataNCCL
