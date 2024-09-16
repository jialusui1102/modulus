# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import matplotlib.pyplot as plt
import json
import numpy as np
import typer
import xarray
import xarray as xr
from scipy.fft import irfft
from scipy.signal import periodogram
# import xskillscore
import torch
import numpy as np
import time
def open_data(file, group=False):
    """
    Opens a dataset from a NetCDF file.

    Parameters:
        file (str): Path to the NetCDF file.
        group (bool, optional): Whether to open the file as a group. Default is False.

    Returns:
        xarray.Dataset: An xarray dataset containing the data from the NetCDF file.
    """
    root = xarray.open_dataset(file)
    root = root.set_coords(["lat", "lon"])
    ds = xarray.open_dataset(file, group=group)
    ds.coords.update(root.coords)
    ds.attrs.update(root.attrs)

    return ds

app = typer.Typer(pretty_exceptions_show_locals=False)

vars = ["10u", "10v", "2t", "refc"]
# files = ["/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_v4_full1_0.nc", "/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_v4_full2_0.nc","/lustre/fsw/coreai_climate_earth2/corrdiff/inferences/twc_mvp_v4_full3_0.nc"]
file = "/code/modulus/examples/generative/corrdiff/outputs/generation/hurricane2022_v2.nc"

@app.command()
def main(output, plot=True, save_data=True, n_ensemble: int = -1, n_timesteps: int = -1):
    """
    Generate and save multiple power spectrum plots from input data.

    Parameters:
        file (str): Path to the input data file.
        output (str): Directory where the generated plots will be saved.

    This function loads and processes various datasets from the input file,
    calculates their power spectra, and generates and saves multiple power spectrum plots.
    The plots include kinetic energy, temperature, and reflectivity power spectra.
    """
    os.makedirs(output, exist_ok=True)

    def savefig(name):
        path = os.path.join(output, name + ".png")
        plt.savefig(path)

    print("n_ensemble", n_ensemble, "n_timesteps", n_timesteps)
    n_members = 32
    plt.figure(figsize=(40,30))
    tic = time.time()
    # for lead in range(9):
    # predictions = []
    # truths = []
    # for file in files:
    #     prediction = open_data(file, group="prediction")
    #     predictions.append(prediction.isel(forecast=lead))
    #     truth = open_data(file, group="truth")
    #     truths.append(truth.isel(forecast=lead))
    # print(prediction)
    # predictions = xr.concat(predictions, dim="time")
    # truths = xr.concat(truths, dim="time")

    # print(lead, time.time()-tic,flush=True)
    predictions = open_data(file, group="prediction")
    if n_ensemble > 0:
        predictions = predictions.isel(ensemble=slice(0, n_ensemble))
    truths = open_data(file, group="truth")
    # print("pred is",prediction['10v'].values)
    
    truths = truths.isel(time=slice(0, n_timesteps))
    predictions = predictions.isel(time=slice(0, n_timesteps))
    # print(prediction['10u'])
    hist = torch.zeros((8, n_members+1))
    with torch.no_grad():
        for idx, var in enumerate(vars):           
            print("var is",var)
            # pdb.set_trace()
            # print("prd",prediction[var].values)
            prediction = torch.from_numpy(predictions[var].values).cuda()
            truth = torch.from_numpy(truths[var].values).cuda()
            prediction = prediction[..., ::2, ::2]
            if var in vars[4:]:
                prediction[prediction<0] = 0
                prediction[prediction>1] = 1
            truth = truth[..., ::2, ::2]
            sorted_predictions, _ = torch.sort(prediction, dim=0)
            for member in range(n_members+1):
                if member==0:
                    a = -float('inf')
                    b = sorted_predictions[0]
                elif member==n_members:
                    a = sorted_predictions[-1]
                    b = float('inf')
                else:
                    a = sorted_predictions[member-1]
                    b = sorted_predictions[member]
                hist[idx, member] = torch.sum((truth>=a) & (truth<b)).detach()/truth.numel()
                print(idx, member, torch.sum((truth>=a) & (truth<b))/truth.numel())

        hist = hist.cpu().detach().numpy()
        # pdb.set_trace()

        # print(lead, time.time()-tic,flush=True)

        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.bar(np.arange(n_members+1),hist[i])
        
        plt.tight_layout()
        savefig("rank_hist")

        # if plot:
        #     i = 1
        #     plt.figure()
        #     for field in hist:
        #         plt.subplot(2,2,i)
        #         i += 1
        #         plt.bar(hist['rank'], hist[field])
        #         plt.title(field)
        #     savefig("rank_hist")


if __name__ == "__main__":
    app()