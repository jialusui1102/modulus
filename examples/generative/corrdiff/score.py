#!/usr/bin/env python
"""Master evaluation script

this consumes inferenced outputs
"""
import typer
import glob
import subprocess
import json
import os
import sys
import environment
import netCDF4 as nc


SCORES_FILENAME = "hurricane2022.nc"
inference_root = "/code/modulus/examples/generative/corrdiff/outputs/generation"
score_root = "/code/modulus/examples/generative/corrdiff/outputs/generation_scores"


def save_config_json(path_netcdf, output):
    with nc.Dataset(path_netcdf) as ds:
        with open(output, "w") as f:
            # TODO should save in json not python format in file
            # inf is needed to load the cfg
            inf = "infinity"
            config = eval(ds.cfg)
            json.dump(config, f)


def call_python_script(args):
    return subprocess.check_call([sys.executable, *args])


# def main(run: str, step: int=typer.Option(), inferences: str=environment.inference_root, scores: str= environment.score_root, inference_type: str = "", rank_hist: bool = False):
#     """
#     Example:
#         ./score.py --step 103614 paper_diffusion --inference-type large_validation_2023
#     """
#     if not inference_type:
#         path_netcdf = os.path.join(inferences, run + ".nc")
#         inference_type = "default"
#     else:
#         pattern = os.path.join(inferences, inference_type, run, f"*{step}*.nc")
#         path_netcdf = glob.glob(pattern)[0]

#     dirname = os.path.join(scores, inference_type, run, f"{step}")
#     os.makedirs(dirname, exist_ok=True)
#     typer.echo(f"Saving output to {dirname}", err=True)
#     save_config_json(path_netcdf, os.path.join(dirname, "config.json"))

#     # TODO remove the hardcoded ensemble sizes
#     # I added this to make a more
#     # reasonable scoring.  I manually moved the files to
#     # scores/v1/large_validation_2023_nEns32 -- NDB 5.7
#     call_python_script(["corrdiff/inference_old/compute_fid.py", path_netcdf, os.path.join(dirname, "fid"), "--n-ensemble", "32"])
#     call_python_script(["corrdiff/score_samples_old.py", path_netcdf, os.path.join(dirname, SCORES_FILENAME), "--n-ensemble", "32"])
#     call_python_script(["corrdiff/inference/power_spectra.py", path_netcdf, os.path.join(dirname, "spectra"), "--n-ensemble", "32"])
#     call_python_script(["corrdiff/case_studies.py", path_netcdf,  dirname])
#     if rank_hist:
#         call_python_script(["corrdiff/inference_old/rank_histogram.py", path_netcdf, os.path.join(dirname, "dispersion"), "--n-ensemble", "8", "--n-timesteps", "20"])

def main(run: str, step: int=typer.Option(), inferences: str=inference_root, scores: str= score_root, inference_type: str = "", rank_hist: bool = False):
    """
    Example:
        ./score.py --step 103614 paper_diffusion --inference-type large_validation_2023
        python3 score.py patched_diffusion --step 1156709 
    """
    if not inference_type:
        # path_netcdf = os.path.join(inferences, run + ".nc")
        path_netcdf = os.path.join(inferences, SCORES_FILENAME)
        inference_type = "default"
    else:
        pattern = os.path.join(inferences, inference_type, run, f"*{step}*.nc")
        path_netcdf = glob.glob(pattern)[0]

    dirname = os.path.join(scores, inference_type, run, f"{step}")
    os.makedirs(dirname, exist_ok=True)
    typer.echo(f"Saving output to {dirname}", err=True)
    save_config_json(path_netcdf, os.path.join(dirname, "config.json"))

    # TODO remove the hardcoded ensemble sizes
    # I added this to make a more
    # reasonable scoring.  I manually moved the files to
    # scores/v1/large_validation_2023_nEns32 -- NDB 5.7
    call_python_script(["corrdiff/inference_old/compute_fid.py", path_netcdf, os.path.join(dirname, "fid"), "--n-ensemble", "32"])
    call_python_script(["corrdiff/score_samples_old.py", path_netcdf, os.path.join(dirname, SCORES_FILENAME), "--n-ensemble", "32"])
    call_python_script(["corrdiff/inference/power_spectra.py", path_netcdf, os.path.join(dirname, "spectra"), "--n-ensemble", "32"])
    call_python_script(["corrdiff/case_studies.py", path_netcdf,  dirname])
    if rank_hist:
        call_python_script(["corrdiff/inference_old/rank_histogram.py", path_netcdf, os.path.join(dirname, "dispersion"), "--n-ensemble", "8", "--n-timesteps", "20"])
    

if __name__ == "__main__":
    typer.run(main)