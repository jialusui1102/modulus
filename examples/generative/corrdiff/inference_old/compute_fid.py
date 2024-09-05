"""Compute the FID of each channel separately

Each channel is downsampled to (299, 299), tiled to 3 RGB channels, and then
passed through the inception model. 
"""
# %%
import os
import numpy as np
import xarray
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import scipy
import json
import typer

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

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

def array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(array_to_list(x) for x in obj)
    elif isinstance(obj, list):
        return [array_to_list(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: array_to_list(x) for k, x in obj.items()}
    else:
        return obj

limits = {
    "maximum_radar_reflectivity": (0, 70),
    "temperature_2m": (266, 310),
    "eastward_wind_10m": (-50, 50),
    "northward_wind_10m": (-50, 50),
}

# Load the pre-trained InceptionV3 model
model = models.inception_v3(pretrained=True, transform_input=False)
model.eval()
model.cuda()

resize = transforms.Resize([299, 299])

def get_mean_sigma(data):
    with torch.no_grad():
        arr = torch.tensor(data).cuda()
        arr = resize(arr)
        rgb = torch.tile(arr[:, None], [3, 1, 1])       
        features = model(rgb)

    mean = features.mean(0)
    variance = features.T @ features / data.shape[0]

    return mean.cpu().numpy(), variance.cpu().numpy()


def normalize(data, field):
    a, b = limits[field]
    center = (a + b) / 2
    return 2 * (data - center) / (b - a)


app = typer.Typer(pretty_exceptions_show_locals=False)
@app.command()
def main(file, output, save_data=True, n_ensemble: int = -1):
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

    pred_fields = open_data(file, group="prediction")
    if n_ensemble >= 0:
        pred_fields = pred_fields.isel(ensemble=slice(0, n_ensemble))
    truth_fields = open_data(file, group="truth")

    fid = {}
    for field in limits:
        truth = truth_fields[field].values
        pred = pred_fields[field][0].values

        mu_ref, sigma_ref = get_mean_sigma(normalize(truth, field))
        mu, sigma = get_mean_sigma(normalize(pred, field))
        fid[field] = calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref)

    if save_data:
        fid_data_path = os.path.join(output, "fid.json")

        fid_data_obj = {"fid": fid}
        with open(fid_data_path, "w") as f:
            json.dump(array_to_list(fid_data_obj), f)

if __name__ == "__main__":
    app()