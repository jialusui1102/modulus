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

import matplotlib.pyplot as plt
import json
import numpy as np
import typer
import xarray
from scipy.fft import irfft
from scipy.signal import periodogram
import xskillscore

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

@app.command()
def main(file, output, plot=True, save_data=True, n_ensemble: int = -1, n_timesteps: int = -1):
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
    samples = {}
    prediction = open_data(file, group="prediction")
    if n_ensemble > 0:
        prediction = prediction.isel(ensemble=slice(0, n_ensemble))
    truth = open_data(file, group="truth")

    truth = truth.isel(time=slice(0, n_timesteps))
    prediction = prediction.isel(time=slice(0, n_timesteps))
    
    samples["prediction"] = prediction
    samples["truth"] = truth

    print(samples["prediction"])
    print(samples["truth"])

    hist = xskillscore.rank_histogram(samples["truth"], samples["prediction"], member_dim="ensemble")

    print("dist done")
    print(hist)

    if save_data:
        path = os.path.join(output, "rank_hist.json")
        with open(path, "w") as f:
            json.dump(hist.to_dict(), f)
            
    if plot:
        i = 1
        plt.figure()
        for field in hist:
            plt.subplot(2,2,i)
            i += 1
            plt.bar(hist['rank'], hist[field])
            plt.title(field)
        savefig("rank_hist")


if __name__ == "__main__":
    app()
