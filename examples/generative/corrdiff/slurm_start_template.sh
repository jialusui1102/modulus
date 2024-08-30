#!/bin/bash

readonly _data_root="/lustre/fsw/coreai_climate_earth2/datasets/cwb-diffusions"
readonly _us_data_root="/lustre/fsw/coreai_climate_earth2/datasets/hrrr"
readonly _corrdiff_root="/lustre/fsw/coreai_climate_earth2/eos-corrdiff-dir"
readonly _code_root="/lustre/fsw/coreai_climate_earth2/asui"
# readonly _cont_mounts="${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
readonly _cont_mounts="/lustre:/lustre:ro,${_corrdiff_root}:/corrdiff:rw,${_us_data_root}:/us_data:ro,${_data_root}:/data:ro,${_code_root}:/code:rw"
readonly _cout_image='/lustre/fsw/coreai_climate_earth2/tge/cont_modulus.sqsh'


readonly _cont_name='asuicorrdiff_interactive'

srun -A coreai_climate_earth2\
        -N1\
        -p batch\
        -t 01:00:00\
        -J coreai_climate_earth2-corrdiff:test\
        --ntasks-per-node=1\
    --container-image=${_cout_image}\
    --container-mounts=${_cont_mounts}\
    --container-name=${_cont_name}\
    --pty bash
