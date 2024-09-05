#!/bin/bash
#SBATCH -A coreai_climate_earth2
#SBATCH -J coreai_climate_earth2-corrdiff:patch_generate
#SBATCH -t 02:10:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --dependency=singleton
#SBATCH -o ./sbatch_logs/train/%x_%j.out
#SBATCH -e ./sbatch_logs/train/%x_%j.err
#SBATCH --exclusive

# be sure we define everything
set -euxo pipefail

# readonly _data_root="/lustre/fsw/portfolios/coreai/users/mnabian"
# readonly _code_root="/lustre/fsw/portfolios/coreai/users/nealp"
# readonly _cont_mounts="${_data_root}:/data:ro,${_code_root}:/code:rw"
# readonly _cont_image="/lustre/fsw/portfolios/coreai/users/tge/cont_modulus.sqsh"


readonly _data_root="/lustre/fsw/coreai_climate_earth2/datasets/cwb-diffusions"
readonly _us_data_root="/lustre/fsw/coreai_climate_earth2/datasets/hrrr"
readonly _corrdiff_root="/lustre/fsw/coreai_climate_earth2/eos-corrdiff-dir"
readonly _code_root="/lustre/fsw/coreai_climate_earth2/asui"
# readonly _cont_mounts="${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
readonly _cont_mounts="/lustre:/lustre:ro,${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
readonly _cout_image='/lustre/fsw/coreai_climate_earth2/tge/cont_modulus.sqsh'


readonly _cont_name='corrdiff'

#export BATCH_PER_GPU=2
export NPROC_PER_NODE=8
#export TOTAL_BATCH=$(($BATCH_PER_GPU * $SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))
export TOTAL_GPU=$((${SLURM_JOB_NUM_NODES} * ${NPROC_PER_NODE}))
export TOTALGPU=$(( ${NPROC_PER_NODE} * ${SLURM_NNODES} ))
RUN_CMD="python -u generate.py"
# rsync -av /code/modulus_pb/modulus /usr/local/lib/python3.10/dist-packages/
# run code
echo "Running on hosts: $(echo $(scontrol show hostname))"
# srun --output=container_output_%j.out \
srun --container-name="${_cont_name}" \
     --container-image="${_cout_image}" \
     --container-mounts="${_cont_mounts}" \
     --ntasks="${TOTALGPU}" --ntasks-per-node="${NPROC_PER_NODE}" \
     bash -c "
     ldconfig
     set -x
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK}
     export LOCAL_RANK=\$(( \${WORLD_RANK} % ${NPROC_PER_NODE} ))
     export HDF5_USE_FILE_LOCKING=FALSE
     export CUDNN_V8_API_ENABLED=1
     export HYDRA_FULL_ERROR=1 
     export WANDB_MODE=online
     unset TORCH_DISTRIBUTED_DEBUG
     pip install treelib
     cd /code/modulus/examples/generative/corrdiff
     ${RUN_CMD}"