#!/bin/bash
#SBATCH -A coreai_devtech_all
#SBATCH -J coreai_climate_earth2-corrdiff_train.multi
#SBATCH -t 03:55:59
#SBATCH -p batch
#SBATCH -N 8
#SBATCH --dependency=singleton   # it means, next job will be automatically run
#SBATCH -o ./updated/multi/%x_%j.out
#SBATCH -e ./updated/multi/%x_%j.err
# be sure we define everything
set -euxo pipefail
# readonly PROJECT_DATA="/lustre/fsw/coreai_climate_earth2/datasets:/datasets:rw"
# readonly _code_root="/lustre/fsw/coreai_climate_earth2/Shoaib"
# readonly _cont_mounts="${PROJECT_DATA}:/data:ro,${_code_root}:/code:rw,/lustre:/lustre:rw"
# readonly _count_image="/lustre/fsw/coreai_climate_earth2/dpruitt/images/modulus.23.11.sqsh"
# readonly _cont_image="/lustre/fsw/coreai_climate_earth2/dpruitt/images/modulus.23.11.sqsh"
# readonly _cont_name="corrdiff"

readonly _data_root="/lustre/fsw/coreai_climate_earth2/datasets/cwb-diffusions"
readonly _us_data_root="/lustre/fsw/coreai_climate_earth2/datasets/hrrr"
readonly _corrdiff_root="/lustre/fsw/coreai_climate_earth2/eos-corrdiff-dir"
readonly _code_root="/lustre/fsw/coreai_climate_earth2/asui"
# readonly _cont_mounts="${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
readonly _cont_mounts="/lustre:/lustre:ro,${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
# readonly _cout_image='/lustre/fsw/coreai_climate_earth2/tge/cont_modulus.sqsh'
readonly _cont_image="/lustre/fsw/coreai_climate_earth2/dpruitt/images/modulus.23.11.sqsh"



readonly _cont_name='corrdiff'
#export BATCH_PER_GPU=2
export NPROC_PER_NODE=8
#export TOTAL_BATCH=$(($BATCH_PER_GPU * $SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))
export TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))
RUN_CMD="torchrun \
--max_restarts 0 \
--nnodes ${SLURM_JOB_NUM_NODES} \
--nproc_per_node ${NPROC_PER_NODE} \
--rdzv_id=${SLURM_JOB_ID} \
--rdzv_backend=c10d \
--rdzv_endpoint=$(hostname) \
train.py"
# run code
echo "Running on hosts: $(echo $(scontrol show hostname))"
srun --container-name="${_cont_name}" \
     --container-image="${_cont_image}" \
     --container-mounts="${_cont_mounts}" \
     bash -c "
     ldconfig
     set -x
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK}
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
     unset TORCH_DISTRIBUTED_DEBUG
     pip install treelib
     rsync -av /code/modulus/modulus/ /usr/local/lib/python3.10/dist-packages/modulus
     cd /code/modulus/examples/generative/corrdiff
     ${RUN_CMD}"