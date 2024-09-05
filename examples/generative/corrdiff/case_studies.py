#!/usr/bin/env python
"""Master evaluation script

this consumes inferenced outputs
"""
import sys
import json
import os
import xarray
import re
import numpy as np 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src/corrdiff')))
from tropical_cyclones_analysis import axi_symmetric_section
from weather_front_analysis import get_mean_cross_section, rotated_winds
sys.path.append('2023-arxiv/figures')
from analysis_utils import load_windspeed


CASES_FILENAME = "case_studies.nc"

def get_config(ds):
    config_str = ds.cfg
    replacements = {
        "None": "null",
        "True": "true",
        "False": "false",
        "inf": "null",
        "'": '"',
    }
    for key, value in replacements.items():
        config_str = config_str.replace(key, value)
    config = json.loads(config_str)
    return config


def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def dataset_to_dict(ds):
    data_dict = {}
    for var in ds.variables:
        # Extract data for each variable and convert to list
        data_dict[var] = {
            "data": numpy_to_list(ds[var].values),
            "coords": {coord: numpy_to_list(ds[coord].values) for coord in ds[var].dims if coord in ds.coords}
        }
    return data_dict


def get_weather_systems(ds, lat, lon, times):
    # TODO: check that the lat lon range include the  even as we move to more data 
    max_latitude = ds.lat.values.max()
    min_latitude = ds.lat.values.min()
    max_longitude = ds.lon.values.max()
    min_longitude = ds.lon.values.min()
    march_2023 = ['2023-03-17T04:00:00', '2023-03-17T05:00:00', '2023-03-17T06:00:00', '2023-03-17T07:00:00']
    jan_2023 = ['2023-01-12T03:00:00', '2023-01-12T04:00:00', '2023-01-12T05:00:00', '2023-01-12T06:00:00']
    if set(times).intersection(set(march_2023)):
        target_lon_start, target_lat_start = 120.11612, 27.946814
        target_lon_end, target_lat_end = 125.39911, 24.142233
        event_times = march_2023
        print("March 2023 weather front found in the data")
    elif set(times).intersection(set(jan_2023)):
        target_lon_start, target_lat_start = 120.32172, 26.103537-0.1
        target_lon_end, target_lat_end = 123.43176, 26.064405-0.1
        event_times = jan_2023
        print("Jan 2023 weather front found in the data")

    else:
        return None, None, None, None, None
    intersection = list(set(event_times) & set(times))
    even_indices = [i for i, t in enumerate(event_times) if t in intersection]
    event_times_indices = [i for i, t in enumerate(times) if t in intersection]
    return target_lon_start, target_lat_start, target_lon_end, target_lat_end, event_times_indices


def analyze_front(path_netcdf):
    ds = xarray.open_dataset(path_netcdf)
    config = get_config(ds)
    lat  = np.array(ds.variables['lat'].values)
    lon  = np.array(ds.variables['lon'].values)
    target_lon_start, target_lat_start, target_lon_end, target_lat_end, event_times_indices = get_weather_systems(ds, lat, lon, config["times"])
    if target_lon_start is None:
        print("No front found in the data")
        return None

    ds = ds.isel(time = event_times_indices)
    ds_prediction = xarray.open_dataset(path_netcdf,group='prediction').isel(time = event_times_indices)
    ds_truth = xarray.open_dataset(path_netcdf,group='truth').isel(time = event_times_indices)
    ds_input = xarray.open_dataset(path_netcdf,group='input').isel(time = event_times_indices)
    skip = 40
    nlevels = 20

    # Initialize lists for storing data along time dimension
    temp_input_data = []
    u_wind_input_data = []
    v_wind_input_data = []
    u_rotated_input_data = []
    v_rotated_input_data = []
    distance_input_t2m_data = []
    values_input_t2m_data = []
    distance_input_u_data = []
    values_input_u_data = []
    distance_input_v_data = []
    values_input_v_data = []

    temp_pred_data = []
    u_wind_pred_data = []
    v_wind_pred_data = []
    u_rotated_pred_data = []
    v_rotated_pred_data = []
    distance_pred_t2m_data = []
    values_pred_t2m_data = []
    distance_pred_u_data = []
    values_pred_u_data = []
    distance_pred_v_data = []
    values_pred_v_data = []

    temp_truth_data = []
    u_wind_truth_data = []
    v_wind_truth_data = []
    u_rotated_truth_data = []
    v_rotated_truth_data = []
    distance_truth_t2m_data = []
    values_truth_t2m_data = []
    distance_truth_u_data = []
    values_truth_u_data = []
    distance_truth_v_data = []
    values_truth_v_data = []

    datetime_values = []

    for time_inx in range(len(event_times_indices)):
        time = config["times"][time_inx]
        datetime_values.append(time)
        u_rotated_truth, v_rotated_truth = rotated_winds(lat, lon, ds_truth.isel(time=time_inx), target_lat_start, target_lon_end, target_lat_end)
        u_rotated_input, v_rotated_input = rotated_winds(lat, lon, ds_input.isel(time=time_inx), target_lat_start, target_lon_end, target_lat_end)
        u_rotated_pred, v_rotated_pred = rotated_winds(lat, lon, ds_prediction.isel(ensemble=0, time=time_inx), target_lat_start, target_lon_end, target_lat_end)
        
        temp_input = ds_input.isel(time=time_inx)['temperature_2m'].values
        temp_pred = ds_prediction.isel(ensemble=0, time=time_inx)['temperature_2m'].values
        temp_truth = ds_truth.isel(time=time_inx)['temperature_2m'].values

        u_wind_input = ds_input.isel(time=time_inx)['eastward_wind_10m'].values
        v_wind_input = ds_input.isel(time=time_inx)['northward_wind_10m'].values

        u_wind_pred = ds_prediction.isel(ensemble=0, time=time_inx)['eastward_wind_10m'].values
        v_wind_pred = ds_prediction.isel(ensemble=0, time=time_inx)['northward_wind_10m'].values

        u_wind_truth = ds_truth.isel(time=time_inx)['eastward_wind_10m'].values
        v_wind_truth = ds_truth.isel(time=time_inx)['northward_wind_10m'].values

        # Collect cross-section data
        distance_pred_t2m, values_pred_t2m = get_mean_cross_section(lat, lon, temp_pred, [448,448], target_lat_start, target_lon_end, target_lat_end)
        distance_truth_t2m, values_truth_t2m = get_mean_cross_section(lat, lon, temp_truth, [448,448], target_lat_start, target_lon_end, target_lat_end)
        distance_input_t2m, values_input_t2m = get_mean_cross_section(lat, lon, temp_input, [448,448], target_lat_start, target_lon_end, target_lat_end)

        distance_pred_u, values_pred_u = get_mean_cross_section(lat, lon, u_rotated_pred, [448,448], target_lat_start, target_lon_end, target_lat_end)
        distance_truth_u, values_truth_u = get_mean_cross_section(lat, lon, u_rotated_truth, [448,448], target_lat_start, target_lon_end, target_lat_end)
        distance_input_u, values_input_u = get_mean_cross_section(lat, lon, u_rotated_input, [448,448], target_lat_start, target_lon_end, target_lat_end)

        distance_pred_v, values_pred_v = get_mean_cross_section(lat, lon, v_rotated_pred, [448,448], target_lat_start, target_lon_end, target_lat_end)
        distance_truth_v, values_truth_v = get_mean_cross_section(lat, lon, v_rotated_truth, [448,448], target_lat_start, target_lon_end, target_lat_end)
        distance_input_v, values_input_v = get_mean_cross_section(lat, lon, v_rotated_input, [448,448], target_lat_start, target_lon_end, target_lat_end)

        temp_input_data.append(ds_input.isel(time=time_inx)['temperature_2m'].values)
        u_wind_input_data.append(ds_input.isel(time=time_inx)['eastward_wind_10m'].values)
        v_wind_input_data.append(ds_input.isel(time=time_inx)['northward_wind_10m'].values)
        u_rotated_input_data.append(u_rotated_input)
        v_rotated_input_data.append(v_rotated_input)

        values_input_t2m_data.append(values_input_t2m)
        values_input_u_data.append(values_input_u)
        values_input_v_data.append(values_input_v)

        temp_pred_data.append(ds_prediction.isel(ensemble=0, time=time_inx)['temperature_2m'].values)
        u_wind_pred_data.append(ds_prediction.isel(ensemble=0, time=time_inx)['eastward_wind_10m'].values)
        v_wind_pred_data.append(ds_prediction.isel(ensemble=0, time=time_inx)['northward_wind_10m'].values)
        u_rotated_pred_data.append(u_rotated_pred)
        v_rotated_pred_data.append(v_rotated_pred)
        
        values_pred_t2m_data.append(values_pred_t2m)
        values_pred_u_data.append(values_pred_u)
        values_pred_v_data.append(values_pred_v)

        temp_truth_data.append(ds_truth.isel(time=time_inx)['temperature_2m'].values)
        u_wind_truth_data.append(ds_truth.isel(time=time_inx)['eastward_wind_10m'].values)
        v_wind_truth_data.append(ds_truth.isel(time=time_inx)['northward_wind_10m'].values)
        u_rotated_truth_data.append(u_rotated_truth)
        v_rotated_truth_data.append(v_rotated_truth)
        
        values_truth_t2m_data.append(values_truth_t2m)
        values_truth_u_data.append(values_truth_u)
        values_truth_v_data.append(values_truth_v)
        

    ds_data = xarray.Dataset({
        "temperature_2m_input": (["time", "y", "x"], np.array(temp_input_data)),
        "u_wind_10m_input": (["time", "y", "x"], np.array(u_wind_input_data)),
        "v_wind_10m_input": (["time", "y", "x"], np.array(v_wind_input_data)),
        "u_rotated_wind_10m_input": (["time", "y", "x"], np.array(u_rotated_input_data)),
        "v_rotated_wind_10m_input": (["time", "y", "x"], np.array(v_rotated_input_data)),

        "values_input_t2m": (["time", "distance"], np.array(values_input_t2m_data)),
        "values_input_u": (["time", "distance"], np.array(values_input_u_data)),
        "values_input_v": (["time", "distance"], np.array(values_input_v_data)),

        "temperature_2m_pred": (["time", "y", "x"], np.array(temp_pred_data)),
        "u_wind_10m_pred": (["time", "y", "x"], np.array(u_wind_pred_data)),
        "v_wind_10m_pred": (["time", "y", "x"], np.array(v_wind_pred_data)),
        "u_rotated_wind_10m_pred": (["time", "y", "x"], np.array(u_rotated_pred_data)),
        "v_rotated_wind_10m_pred": (["time", "y", "x"], np.array(v_rotated_pred_data)),

        "values_pred_t2m": (["time", "distance"], np.array(values_pred_t2m_data)),
        "values_pred_u": (["time", "distance"], np.array(values_pred_u_data)),
        "values_pred_v": (["time", "distance"], np.array(values_pred_v_data)),

        "temperature_2m_truth": (["time", "y", "x"], np.array(temp_truth_data)),
        "u_wind_10m_truth": (["time", "y", "x"], np.array(u_wind_truth_data)),
        "v_wind_10m_truth": (["time", "y", "x"], np.array(v_wind_truth_data)),
        "u_rotated_wind_10m_truth": (["time", "y", "x"], np.array(u_rotated_truth_data)),
        "v_rotated_wind_10m_truth": (["time", "y", "x"], np.array(v_rotated_truth_data)),
        
        "values_truth_t2m": (["time", "distance"], np.array(values_truth_t2m_data)),
        "values_truth_u": (["time", "distance"], np.array(values_truth_u_data)),
        "values_truth_v": (["time", "distance"], np.array(values_truth_v_data)),
    },
    coords={
        "time": datetime_values,
        "lat": (["y", "x"], lat),
        "lon": (["y", "x"], lon),
        "distance": (["distance"], distance_truth_u)
    })

    json_front = dataset_to_dict(ds_data)
    return json_front


def get_tropical_cyclones(ds, lat, lon, times):
    # TODO: check that the lat lon range include the  even as we move to more data 
    max_latitude = ds.lat.values.max()
    min_latitude = ds.lat.values.min()
    max_longitude = ds.lon.values.max()
    min_longitude = ds.lon.values.min()
    time_Haikui2023 = [
        '2023-09-02T11:00:00','2023-09-02T12:00:00','2023-09-02T13:00:00','2023-09-02T14:00:00',
        '2023-09-02T15:00:00','2023-09-02T16:00:00','2023-09-02T17:00:00','2023-09-02T18:00:00',
        '2023-09-02T19:00:00','2023-09-02T20:00:00','2023-09-02T21:00:00','2023-09-02T22:00:00',
        '2023-09-02T23:00:00','2023-09-03T00:00:00','2023-09-03T01:00:00','2023-09-03T02:00:00',
        '2023-09-03T03:00:00','2023-09-03T04:00:00','2023-09-03T05:00:00','2023-09-03T06:00:00',
        ]
    if set(times).intersection(set(time_Haikui2023)):
        print("Haikui2023 typhoon found in the data")
        event_times = time_Haikui2023
        true_lon = np.array([124.3, 124.2, 124.1, 124.0, 123.9, 123.7, 123.5, 123.4, 123.2,
                    123.1, 123.0, 123.0, 122.9, 122.8, 122.7, 122.5, 122.3, 122.1, 122.0, 121.9, 121.7, 121.5])
        true_lat = np.array([22.3,22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.3, 22.4, 22.5,
                                22.5, 22.6, 22.6, 22.7, 22.8, 22.8, 22.9, 23.0, 23.0, 23.0, 23.0])
    else:
        return np.array([]),np.array([]), np.array([])
    
    intersection = list(set(event_times) & set(times))
    even_indices = [i for i, t in enumerate(event_times) if t in intersection]
    event_times_indices = [i for i, t in enumerate(times) if t in intersection]
    return true_lat[even_indices], true_lon[even_indices], event_times_indices


def analyze_tropical_cyclones(path_netcdf):
    ens = 0 # ensmeble member to save 2D map
    L = 80  # saving +/- L pixels aroud the storm from the 2D map
    ds = xarray.open_dataset(path_netcdf)
    config = get_config(ds)
    lat  = np.array(ds.variables['lat'].values)
    lon  = np.array(ds.variables['lon'].values)
    true_lat_list, true_lon_list, event_times_indices = get_tropical_cyclones(ds, lat, lon, config["times"])
    if true_lat_list.size==0:
        print("No typhoon found in the data")
        return None, None

    ds = ds.isel(time = event_times_indices)
    lat  = np.array(ds.variables['lat'])
    lon  = np.array(ds.variables['lon'])
    datetime_values = np.array(ds.time, dtype='datetime64[ns]')
    dx = dy = round((ds.lat[1,1]-ds.lat[0,1]).values*111.1)*1000.0

    pred = xarray.open_dataset(path_netcdf, group = "prediction").isel(time = event_times_indices)
    truth = xarray.open_dataset(path_netcdf, group = "truth").isel(time = event_times_indices)
    era5 = xarray.open_dataset(path_netcdf, group = "input").isel(time = event_times_indices)
    diffusion_windspeed = load_windspeed(pred)
    wrf_windspeed = load_windspeed(truth)
    era5_windspeed = load_windspeed(era5)

    fix_bin_edges = np.linspace(0.0, 60.0, 30)
    combined_values_input = np.zeros(len(fix_bin_edges) - 1)
    combined_values_pred = np.zeros(len(fix_bin_edges) - 1)

    diffusion_reflectivity_data = []
    wrf_reflectivity_data = []
    pdf_data = {
        "WRF": {"bin_centers": [], "values": []},
        "ERA5": {"bin_centers": [], "values": []},
        "Diffusion": {"bin_centers": [], "values": []}
    }
    radius_km_data = []
    windspeed_data = {
        "WRF": [],
        "ERA5": [],
        "mean_diffusion": [],
        "stddev_diffusion": []
    }

    radii = np.linspace(0, 200, 201) * dx
    r = radii / 1000.0

    for tc_time in range(len(true_lat_list)):
        true_lat = true_lat_list[tc_time]
        true_lon = true_lon_list[tc_time]
        V_wrf, i_c, j_c = axi_symmetric_section(
                            radii,
                            lon,
                            lat,
                            truth["eastward_wind_10m"][tc_time,:,:].values,
                            truth["northward_wind_10m"][tc_time,:,:].values,
                            true_lon,
                            true_lat,
                            )
        
        V_era5, _, _  = axi_symmetric_section(
                            radii,
                            lon,
                            lat,
                            era5["eastward_wind_10m"][tc_time,:,:].values,
                            era5["northward_wind_10m"][tc_time,:,:].values,
                            true_lon,
                            true_lat,
                            )
        
        n_samples = np.shape(pred["eastward_wind_10m"])[0]
        V_diffusion = np.zeros((len(radii)-1, n_samples))
        for i in range(n_samples):
            V_diffusion[:,i], _, _  = axi_symmetric_section(
                                    radii,
                                    lon,
                                    lat,
                                    pred["eastward_wind_10m"][i,tc_time,:,:].values,
                                    pred["northward_wind_10m"][i,tc_time,:,:].values,
                                    true_lon,
                                    true_lat,
                                    )

        mean_V_diffusion = np.mean(V_diffusion, axis=1)
        std_V_diffusion = np.std(V_diffusion, axis=1)

        pdf_values_era5_tmp, _ = np.histogram(era5_windspeed[tc_time].values.flatten(), bins=fix_bin_edges)
        pdf_values_diffusion_tmp, _ = np.histogram(diffusion_windspeed[:, tc_time].values.flatten(), bins=fix_bin_edges)
        combined_values_input += pdf_values_era5_tmp
        combined_values_pred += pdf_values_diffusion_tmp

        i_c_min = np.max([i_c - L, 0])
        j_c_min = np.max([j_c - L, 0])
        i_c_max = np.min([i_c + L, 447])
        j_c_max = np.min([j_c + L, 447])
        
        bins = np.linspace(0,50,100)
        pdf_values_era5, bin_edges_input = np.histogram(era5_windspeed[tc_time, i_c_min:i_c_max, j_c_min:j_c_max].values.flatten(), bins=bins, density=True)
        bin_centers_era5 = 0.5 * (bin_edges_input[1:] + bin_edges_input[:-1])
        pdf_values_diffusion, bin_edges_diffusion = np.histogram(diffusion_windspeed[:, tc_time, i_c_min:i_c_max, j_c_min:j_c_max].values.flatten(), bins=bins, density=True)
        bin_centers_diffusion = 0.5 * (bin_edges_diffusion[1:] + bin_edges_diffusion[:-1])
        pdf_values_wrf, bin_edges_wrf = np.histogram(wrf_windspeed[tc_time, i_c_min:i_c_max, j_c_min:j_c_max].values.flatten(), bins=bins, density=True)
        bin_centers_wrf = 0.5 * (bin_edges_wrf[1:] + bin_edges_wrf[:-1])

        # Append data to lists
        diffusion_reflectivity_data.append(pred.maximum_radar_reflectivity.isel(ensemble=ens).values[tc_time])
        wrf_reflectivity_data.append(truth.maximum_radar_reflectivity.values[tc_time])
        
        pdf_data["WRF"]["bin_centers"].append(bin_centers_wrf)
        pdf_data["WRF"]["values"].append(pdf_values_wrf)
        pdf_data["ERA5"]["bin_centers"].append(bin_centers_era5)
        pdf_data["ERA5"]["values"].append(pdf_values_era5)
        pdf_data["Diffusion"]["bin_centers"].append(bin_centers_diffusion)
        pdf_data["Diffusion"]["values"].append(pdf_values_diffusion)

        windspeed_data["WRF"].append(V_wrf)
        windspeed_data["ERA5"].append(V_era5)
        windspeed_data["mean_diffusion"].append(mean_V_diffusion)
        windspeed_data["stddev_diffusion"].append(std_V_diffusion)
        
    radius_km_data = np.array(r[0:-1])
    pdf_values_wrf = np.array(pdf_data["WRF"]["values"])
    bin_centers_wrf = np.array(pdf_data["WRF"]["bin_centers"][0])
    pdf_values_era5 = np.array(pdf_data["ERA5"]["values"])
    bin_centers_era5 = np.array(pdf_data["ERA5"]["bin_centers"][0])
    pdf_values_diffusion = np.array(pdf_data["Diffusion"]["values"])
    bin_centers_diffusion = np.array(pdf_data["Diffusion"]["bin_centers"][0])
    ds_windspeed = xarray.Dataset({
        "wrf_windspeed": (("time", "radius"), windspeed_data["WRF"]),
        "era5_windspeed": (("time", "radius"), windspeed_data["ERA5"]),
        "mean_diffusion_windspeed": (("time", "radius"), windspeed_data["mean_diffusion"]),
        "stddev_diffusion_windspeed": (("time", "radius"), windspeed_data["stddev_diffusion"]),
        "pdf_values_wrf": (("time", "bins"), pdf_values_wrf),
        "pdf_values_era5": (("time", "bins"), pdf_values_era5),
        "pdf_values_diffusion": (("time", "bins"), pdf_values_diffusion),
    }, coords={
        "time": datetime_values,
        "radius": radius_km_data,
        "bins": bin_centers_wrf
    })

    ds_radar = xarray.Dataset({
        "latitude": (["y", "x"], lat),
        "longitude": (["y", "x"], lon),
        "diffusion_reflectivity": (["time", "y", "x"], np.array(diffusion_reflectivity_data)),
        "wrf_reflectivity": (["time", "y", "x"], np.array(wrf_reflectivity_data))
    }, coords={
        "time": datetime_values,
        "lat": (["y", "x"], lat),
        "lon": (["y", "x"], lon)
    })
    
    radar_dict = dataset_to_dict(ds_radar)
    for key in windspeed_data:
        windspeed_data[key] = np.array(windspeed_data[key])
    windspeed_dict = dataset_to_dict(ds_windspeed)
    return windspeed_dict, radar_dict


def main(path_netcdf, dirname):
    ds = xarray.open_dataset(path_netcdf)
    windspeed_dict, radar_dict = analyze_tropical_cyclones(path_netcdf)
    dirname = "./"
    if windspeed_dict is not None:
        # save TC radar data
        radar_filename = os.path.join(dirname, "radar.json")
        with open(radar_filename, 'w') as f:
            json.dump(radar_dict, f)
        print(f"JSON data saved to {radar_filename}")
        # save TC windspeed data
        windspeed_filename = os.path.join(dirname, "windspeed.json")
        with open(windspeed_filename, 'w') as f:
            json.dump(windspeed_dict, f)
        print(f"JSON data saved to {windspeed_filename}")
    

    front_dict = analyze_front(path_netcdf)
    if front_dict is not None:
        front_filename = os.path.join(dirname, "front.json")
        with open(front_filename, 'w') as f:
            json.dump(front_dict, f)
        print(f"JSON data saved to {front_filename}")
    
if __name__ == "__main__":
    # here I am cathcing inputs from sys subprocess call
    #  which uses 0 for the script name
    path_netcdf = sys.argv[1]
    dirname = sys.argv[2]
    main(path_netcdf, dirname)