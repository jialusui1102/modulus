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
import argparse
import cftime
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr
import cartopy 
import cartopy.crs as ccrs
import imageio
import datetime
import tqdm

def time_range(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step: datetime.timedelta,
    inclusive: bool = False,
):
    """Like the Python `range` iterator, but with datetimes."""
    t = start_time
    while (t <= end_time) if inclusive else (t < end_time):
        yield t
        t += step


def pattern_correlation(x, y):
    """Pattern correlation"""
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.mean((x - mx) ** 2)
    vy = np.mean((y - my) ** 2)

    a = np.mean((x - mx) * (y - my))
    b = np.sqrt(vx * vy)

    return a / b


def plot_channels(group, time_idx: int):
    """Plot channels"""
    # weather sub-plot
    num_channels = len(group.variables)
    ncols = 4
    fig, axs = plt.subplots(
        nrows=(
            num_channels // ncols
            if num_channels % ncols == 0
            else num_channels // ncols + 1
        ),
        ncols=ncols,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        figsize=(15, 15),
    )

    for ch, ax in zip(sorted(group.variables), axs.flat):
        # label row
        x = group[ch][time_idx]
        ax.set_title(ch)
        ax.imshow(x)


def channel_eq(a, b):
    """Check if two channels are equal in variable and pressure."""
    variable_equal = a["variable"] == b["variable"]
    pressure_is_nan = np.isnan(a["pressure"]) and np.isnan(b["pressure"])
    pressure_equal = a["pressure"] == b["pressure"]
    return variable_equal and (pressure_equal or pressure_is_nan)


def channel_repr(channel):
    """Return a string representation of a channel with variable and pressure."""
    v = channel["variable"]
    pressure = channel["pressure"]
    return f"{v}\n Pressure: {pressure}"

def get_clim(groups):
    """Get color limits (clim) for output channels based on prediction and truth data."""
    vmin = min([i.min() for i in groups])
    vmax = max([i.max() for i in groups])
    colorlimits = (vmin, vmax)
    return colorlimits


# Create the parser
parser = argparse.ArgumentParser()

# Add the positional arguments
# parser.add_argument("file", help="Path to the input file")
# parser.add_argument("output_dir", help="Path to the output directory")

# Add the optional argument
parser.add_argument("--sample", help="Sample to plot", default=0, type=int)

# Parse the arguments
args = parser.parse_args()


def main():
    """Plot single sample"""
    file = "/us_data/downscaling/hrrr/2022.zarr"
    res_url_base = "../image_outdir_hrrr_reslossv2_9M_casestudy_2_0.nc"
    
    # txt_msg = "reg_ckpt:/code/hrrr_reg/training-state-regression-000534.mdlus"
    # txt_msg = "reg_ckpt:/code/hrrr_reg/training-state-regression-000534.mdlus" + "\n" + "res_ckpt:/code/modulus_hrrr/examples/generative/corrdiff/output_diffusion/training-state-diffusion-008405.mdlus"
    output_dir = os.path.splitext(__file__)[0]
    os.makedirs(output_dir, exist_ok=True)

    time_format = "%Y-%m-%dT%H:%M:%S"

    f = xr.open_zarr(file)
    lon = f['longitude'].isel(y=slice(1,1057)).isel(x=slice(4,1796))
    lat = f['latitude'].isel(y=slice(1,1057)).isel(x=slice(4,1796))

    # t = "2022-07-25T00:00:00"
    # t = "2022-09-29T00:00:00"
    # t = "2022-01-28T00:00:00"
    # t = "2022-02-04T00:00:00"
    # times = ["2022-02-03T12:00:00", "2022-04-06T12:00:00", "2022-11-05T00:00:00", "2022-09-25T12:00:00", "2022-09-29T00:00:00"]
    times = ["2022-11-03T04:00:00", "2022-01-18T19:00:00", "2022-06-25T03:00:00", "2022-02-08T23:00:00"]

    times_range = None
    times_range = ["2022-11-07T12:00:00", "2022-11-11T12:00:00"]

    if times_range is not None:
        times = []
        t_start = datetime.datetime.strptime(times_range[0], time_format)
        t_end = datetime.datetime.strptime(times_range[1], time_format)
        dt = datetime.timedelta(hours=(times_range[2] if len(times_range) > 2 else 1))
        times = [
            t.strftime(time_format)
            for t in time_range(t_start, t_end, dt, inclusive=True)
        ]

    # start_hours = (np.datetime64(t) - np.datetime64("2022-01-01T01:00:00")) // np.timedelta64(1, 'h')

    # f = f['HRRR'].sel(time=t).isel(y=slice(1,1057)).isel(x=slice(4,1796))
    res = xr.open_dataset(res_url_base, group="prediction").isel(ensemble=0)
    gt = xr.open_dataset(res_url_base, group="truth")
    inp = xr.open_dataset(res_url_base, group="input")
    
    lon = lon.load()
    lat = lat.load()

    group_label = ["Ground truth", "Patch-diff", "ERA5 Input"]

    kf ={"refc":"maximum_radar_reflectivity", 
        "10u": "eastward_wind_10m", 
        "10v": "northward_wind_10m", 
        "2t": "temperature_2m"}

    key2inp = {"10u": "u10m",
                "10v": "v10m",
                "2t": "t2m"}

    for channel in ['2t', '10u', '10v', 'refc']:
        # label row
        inp_channel = key2inp.get(channel, None)
        res_c = res[channel]
        gt_c = gt[channel]
        inp_c = inp[inp_channel] if inp_channel is not None else gt_c
        vmin, vmax = get_clim([res_c, gt_c, inp_c])
        for t_idx, t in enumerate(times):
            fs = [gt_c.isel(time=t_idx).load(), res_c.isel(time=t_idx).load(), inp_c.isel(time=t_idx).load()]
        

            # projection = ccrs.LambertConformal(
            #     central_longitude = -97.5,
            #     central_latitude = 38.5,
            #     standard_parallels = (38.5,38.5),
            #     cutoff = 0)

            fig, axss = plt.subplots(
                nrows=3,
                ncols=1,
                sharex=True,
                sharey=True,
                constrained_layout=True,
                figsize=(14, 16),
                subplot_kw=dict(projection=ccrs.PlateCarree())
            )
            for row, group in enumerate(fs):
                axs = axss[row]
                gl = axs.gridlines(
                    crs=ccrs.PlateCarree(),
                    color="black",
                    alpha=0.0,
                    draw_labels=True,
                    linestyle="None",
                )
                axs.coastlines(linewidth=0.5, color="white")
                axs.set_title(f"{group_label[row]}")

                def plot_panel(ax, data, **kwargs):
                    if channel == "refc":
                        return ax.pcolormesh(
                            lon, lat, data, cmap="magma", vmin=0, vmax=vmax, # transform=ccrs.PlateCarree()
                        )
                    if channel == "2t":
                        return ax.pcolormesh(
                            lon, lat, data, cmap="magma", vmin=vmin, vmax=vmax, # transform=ccrs.PlateCarree()
                        )
                    else:
                        if vmin < 0 < vmax:
                            bound = max(abs(vmin), abs(vmax))
                            vmin1 = -bound
                            vmax1 = bound
                        else:
                            vmin1 = vmin
                            vmax1 = vmax
                        return ax.pcolormesh(
                            lon, lat, data, cmap="RdBu_r", vmin=vmin1, vmax=vmax1, #transform=ccrs.PlateCarree()
                        )


                im = plot_panel(axs, group)
                axs.set_xlabel("longitude [deg E]")
                axs.set_ylabel("latitude [deg N]")
            
            fig.suptitle(f'{kf[channel]} of CONUS at {t}', fontsize=16)
            fig.subplots_adjust(top=0.88)
            cb = plt.colorbar(im, ax=axss.ravel().tolist())
            cb.set_label(kf[channel])
            plt.savefig(f"{output_dir}/Patch_diff-{channel}-{t}.png")
            plt.close()

    

    writer = imageio.get_writer('radar_reflectivity_movie_new.mp4', fps = 3)
    dirFiles = os.listdir('./plot_samples/') #list of directory files
    # dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dirFiles.sort()
    for im in dirFiles:
        writer.append_data(imageio.imread('./plot_samples/'+im))
    writer.close()

if __name__ == "__main__":
    main()
