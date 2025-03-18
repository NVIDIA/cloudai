. /etc/os-release
hpcx_version=${HPCX_DIR##*/}

while IFS=: read -r value; do
    echo "$value"
done <<EOF
[system]
user = ${USER:-$(whoami)}
system_name = "${SYSTEM:-$(command -v sacctmgr &> /dev/null && sacctmgr -nP show cluster format=cluster || echo null)}"
os_type = "${ID:-null}"
os_version = "${VERSION:-null}"
linux_kernel_version = "$(uname -r)"
gpu_arch_type = "$(nvidia-smi -q 2>/dev/null | grep "Product Name" | head -n1 | awk -F': ' '{print $2}' || echo null)"
cpu_model_name = "$(lscpu 2>/dev/null | grep "Model name:" | sed 's/Model name:[[:space:]]*//' || echo null)"
cpu_arch_type = "$(lscpu 2>/dev/null | grep "Architecture:" | sed 's/Architecture:[[:space:]]*//' || echo null)"

[mpi]
mpi_type = "$(mpirun --version 2>/dev/null | grep -i "open mpi" -q && echo openmpi || echo null)"
mpi_version = "$(mpirun --version 2>/dev/null | grep -oP "(?<=\(Open MPI\) )[^\s]+$" || echo null)"
hpcx_version = "${hpcx_version:-null}"

[cuda]
cuda_build_version = "${CUDA_BUILD_VERSION:-null}"
cuda_runtime_version = "$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+(?=,)' || echo null)"
cuda_driver_version = "$(nvidia-smi 2>/dev/null | grep -oP "(?<=Driver Version: )[\d\.]+" || echo null)"

[network]
nics = "$(lspci 2>/dev/null | grep -E "Ethernet controller|Infiniband controller" | grep -v "BlueField" | awk -F": " '/Mellanox/ {found=1;print $NF;exit 0} END {if (!found) exit 1}' || echo null)"
switch_type = "${SWITCH:-null}"
network_name = "${NETWORK:-null}"
mofed_version = "$(command -v ofed_info >/dev/null && ofed_info -s 2>/dev/null | sed 's/:$//' || echo null)"
libfabric_version = "$(command -v fi_info >/dev/null && fi_info --version 2>/dev/null | grep "Libfabric" | awk '{print $2}' || echo null)"

[nccl]
version = "${NCCL_VERSION:-null}"
commit_sha = "${NCCL_COMMIT_SHA:-null}"
EOF