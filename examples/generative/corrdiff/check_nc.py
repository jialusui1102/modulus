import xarray as xr
import pdb
import netCDF4 as nc

# Open the NetCDF file
ds = nc.Dataset('/lustre/fsw/coreai_climate_earth2/nealp/scripts/dummy.nc','r')
ds2 = nc.Dataset('outputs/generation/hurricane2022.nc','r')
# pdb.set_trace()
# Optionally, you can loop over and print each group
for group_name, group in ds.groups.items():
    print(f"Group: {group_name}")
    print(group)
# Display the structure and metadata
# print(ds)

# print(ds.variables)
# Print specific variable data
# print(ds['2t'])
# 2t, 10u, 10v, refc

def open_data(file, group=False):
    """
    Opens a dataset from a NetCDF file.

    Parameters:
        file (str): Path to the NetCDF file.
        group (bool, optional): Whether to open the file as a group. Default is False.

    Returns:
        xarray.Dataset: An xarray dataset containing the data from the NetCDF file.
    """
    root = xr.open_dataset(file)
    root = root.set_coords(["lat", "lon"])
    ds = xr.open_dataset(file, group=group)
    ds.coords.update(root.coords)
    ds.attrs.update(root.attrs)

    return ds
# file = "/lustre/fsw/coreai_climate_earth2/nealp/scripts/dummy.nc"
file = "outputs/generation/hurricane2022_v2.nc"
# pred = open_data('outputs/generation/hurricane2022.nc',group='prediction')
# truth = open_data('outputs/generation/hurricane2022.nc',group='truth')
pred = open_data(file,group='prediction')
truth = open_data(file,group='truth')
# Access a subset of the data (e.g., first time slice, first 10x10 grid)
subset_p = pred['10v'].isel(time=0, y=slice(0, 10), x=slice(0, 10))
subset_t = truth['10v'].isel(time=0, y=slice(0, 10), x=slice(0, 10))
# print(subset.values)  # This will only load the small subset

pdb.set_trace()

