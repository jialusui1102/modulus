#adapted from https://github.com/NVlabs/FourCastNet
#HRRR adaptation by Jaideep Pathak (NVIDIA), Peter Harrington (Lawrence Berkeley National Lab)

from datetime import datetime, timedelta
import glob
import logging
import os
from typing import Iterable, Tuple, Union
import cv2
import s3fs

import dask
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
import xarray as xr

from modulus.distributed import DistributedManager

from .base import ChannelMetadata, DownscalingDataset

from .time import time_range
import nvtx

# channel ordering in the datasets changes based on year
# doing this to avoid breaking things
# some people would say this is a hack, they're right
hrrr_stats_channels = ['2t', '10u', '10v', 'refc', 'gust', '2d', '2r', 'tp', 'cape', 'cin', 'tcc', 'dswrf', 'dlwrf', 'vbdsf', 'veril', 'hpbl']
era5_stats_channels = ['u10m', 'v10m', 't2m', 'tcwv', 'sp', 'msl', 'u1000', 'u850', 'u500', 'u250', 'v1000', 'v850', 'v500', 'v250', 'z1000', 'z850', 'z500', 'z250', 't1000', 't850', 't500', 't250', 'q1000', 'q850', 'q500', 'q250']


def get_dataset(*, data_path, train, **kwargs):
    return HrrrEra5Dataset(
        **kwargs,
        train=train,
        location=data_path,
    )


class HrrrEra5Dataset(DownscalingDataset):
    '''
    Paired dataset object serving time-synchronized pairs of ERA5 and HRRR samples
    Expects data to be stored under directory specified by 'location'
        ERA5 under <root_dir>/era5/
        HRRR under <root_dir>/hrrr/
    '''
    def __init__(
        self,
        *,
        location: str,
        train: bool = True,
        normalize: bool = True,
        dataset_name: str = 'hrrr',
        hrrr_stats_dir: str = 'stats',
        exclude_channels: Iterable[str] = ('2t', 'refc', 'gust', '2d', '2r', 'cape', 'cin', 'tcc', 'dswrf', 'dlwrf', 'vbdsf', 'veril', 'hpbl'),
        out_channels: Iterable[str] = None,
        in_channels: Iterable[str] = None,
        train_years: Iterable[int] = (2018, 2019, 2020, 2021),
        valid_years: Iterable[int] = (2017, 2022),
        hrrr_window: Union[Tuple[Tuple[int, int], Tuple[int, int]], None] = None,
        sample_shape: Tuple[int,int] = (448, 448),
        ds_factor: int = 1,
        shard: bool = False,
        overfit: bool = False,
        # use_all: bool = False,
        all_times: bool = False,
    ):
        dask.config.set(scheduler='synchronous') # for threadsafe multiworker dataloaders
        self.location = location
        self.train = train
        self.normalize = normalize
        self.dataset_name = dataset_name
        self.exclude_channels = list(exclude_channels)
        self.hrrr_channels = out_channels
        self.era5_channels = in_channels
        self.train_years = list(train_years)
        self.valid_years = list(valid_years)
        self.hrrr_window = hrrr_window
        self.sample_shape = sample_shape
        self.ds_factor = ds_factor
        self.shard = shard
        # self.use_all = use_all
        self.all_times = all_times
        self.s3 = s3fs.S3FileSystem() if "s3:" in location else None

        self._get_files_stats()
        self.overfit = overfit

        # expand = False
        # if expand:
        #     # 3.5 million pixels
        #     self.y_expand = 960
        #     self.x_expand = 960
        #     # 5.5 million pixels
        #     # self.y_expand = 1344
        #     # self.x_expand = 608
        #     # 7.5 million pixels
        #     # self.y_expand = 1056
        #     # self.x_expand = 1792
        #     # 9.2 million pixels
        #     # self.y_expand = 1504
        #     # self.x_expand = 1792
        #     # # 10.5 million pixels
        #     # self.y_expand = 1864
        #     # self.x_expand = 1792
        # else:
        #     self.y_expand = 0
        #     self.x_expand = 0

        self.kept_hrrr_channel_names = self._get_hrrr_channel_names()
        # channel to idx varies across files so we only save the channel names
        kept_hrrr_channels = [hrrr_stats_channels.index(x) for x in self.kept_hrrr_channel_names]
        # logging.info(f'means load {kept_hrrr_channels}')
        means_file = os.path.join(self.location, self.dataset_name, hrrr_stats_dir, 'means.npy')
        stds_file = os.path.join(self.location, self.dataset_name, hrrr_stats_dir, 'stds.npy')
        if self.s3:
            print("loading stats from s3")
            means_file = self.s3.open(means_file)
            stds_file = self.s3.open(stds_file)
        self.means_hrrr = np.load(means_file)[kept_hrrr_channels, None, None]
        self.stds_hrrr = np.load(stds_file)[kept_hrrr_channels, None, None]

        # self.means_hrrr2 = xr.open_dataarray(
        #     os.path.join(self.location, self.dataset_name, hrrr_stats_dir, 'means.zarr')
        # ).sel(channel=[self.kept_hrrr_channel_names]).values[:,None,None]
        # self.stds_hrrr2 = xr.open_dataarray(
        #     os.path.join(self.location, self.dataset_name, hrrr_stats_dir, 'stds.zarr')
        # ).sel(channel=[self.kept_hrrr_channel_names]).values[:,None,None]      

        # print(f'hrrr')
        # print(self.kept_hrrr_channel_names)
        # print(kept_hrrr_channels)
        # print(f'loaded means {self.means_hrrr.squeeze()}\nstds {self.stds_hrrr.squeeze()}')

        self.kept_era5_channel_names = self._get_era5_channel_names()
        kept_era5_channels = [era5_stats_channels.index(x) for x in self.kept_era5_channel_names]
        # logging.info(f'means load {kept_era5_channels}')
        means_file = os.path.join(self.location, 'era5', 'stats', 'means.npy')
        stds_file = os.path.join(self.location, 'era5', 'stats', 'stds.npy')
        if self.s3:
            means_file = self.s3.open(means_file)
            stds_file = self.s3.open(stds_file)
        self.means_era5 = np.load(means_file)[kept_era5_channels, None, None]
        self.stds_era5 = np.load(stds_file)[kept_era5_channels, None, None]

        # self.means_era52 = xr.open_dataarray(
        #     os.path.join(self.location, 'era5', 'stats', 'means.zarr')
        # ).sel(channel=[self.kept_er5_channel_names]).values[:,None,None]
        # self.stds_era52 = xr.open_dataarray(
        #     os.path.join(self.location, 'era5', 'stats', 'stds.zarr')
        # ).sel(channel=[self.kept_er5_channel_names]).values[:,None,None]

    def _get_hrrr_channel_names(self):
        if self.hrrr_channels:
            kept_hrrr_channels = [x for x in self.hrrr_channels if x in self.base_hrrr_channels]
            if len(kept_hrrr_channels) != len(self.hrrr_channels):
                print(f'Not all HRRR channels in dataset. Missing {self.hrrr_channels-kept_hrrr_channels}')
        else:
            kept_hrrr_channels = self.base_hrrr_channels

        return list(kept_hrrr_channels)

    def _get_era5_channel_names(self):
        if self.era5_channels:
            kept_era5_channels = [x for x in self.era5_channels if x in self.base_era5_channels]
            if len(kept_era5_channels) != len(self.era5_channels):
                print(f'Not all ERA5 channels in dataset. Missing {self.era5_channels-kept_era5_channels}')
        else:
            kept_era5_channels = self.base_era5_channels

        return list(kept_era5_channels)

    def _get_files_stats(self):
        '''
        Scan directories and extract metadata for ERA5 and HRRR

        Note: This makes the assumption that the lowest numerical year has the 
        correct channel ordering for the means
        '''

        # ERA5 parsing
        # glob all era5 paths under location/era5 and subdirectories
        self.ds_era5 = {}
        if self.s3:
            print("initializing input from s3")
            era5_short_paths = self.s3.glob(os.path.join(self.location.replace("s3://",""),'era5',"????.zarr"))
            era5_paths = ["s3://" + path for path in era5_short_paths]
        else:
            era5_paths = glob.glob(os.path.join(self.location, "era5", "**", "????.zarr"), recursive=True)
        era5_years = [int(os.path.basename(x).replace('.zarr', '')) for x in era5_paths if "stats" not in x]
        self.era5_paths = dict(zip(era5_years, era5_paths))

        # keep only training or validation years
        years = self.train_years if self.train else self.valid_years
        print(f'Training {self.train} years {years}')
        self.era5_paths = {year: path for (year, path) in self.era5_paths.items() if year in years}
        self.n_years = len(self.era5_paths)

        with xr.open_zarr(self.era5_paths[years[0]], consolidated=True) as ds:
            self.base_era5_channels = list(ds.channel.values)
            self.era5_lat = ds.latitude
            self.era5_lon = ds.longitude

        # HRRR parsing
        self.ds_hrrr = {}
        if self.s3:
            print("initializing output from s3")
            hrrr_short_paths = self.s3.glob(os.path.join(self.location.replace("s3://",""),self.dataset_name,"????.zarr"))
            hrrr_paths = ["s3://" + path for path in hrrr_short_paths]
        else:
            hrrr_paths = glob.glob(os.path.join(self.location, self.dataset_name, "**", "????.zarr"), recursive=True)
        hrrr_years = [int(os.path.basename(x).replace('.zarr', '')) for x in hrrr_paths if "stats" not in x]
        self.hrrr_paths = dict(zip(hrrr_years, hrrr_paths))

        # keep only training or validation years
        self.hrrr_paths = {year: path for (year, path) in self.hrrr_paths.items() if year in years}
        self.years = sorted(self.hrrr_paths.keys())

        assert set(era5_years) == set(hrrr_years), 'Number of years for ERA5 in %s and HRRR in %s must match'%(os.path.join(self.location, "era5/*.zarr"),
                                                                                                os.path.join(self.location, "hrrr/*.zarr"))
        with xr.open_zarr(self.hrrr_paths[years[0]], consolidated=True) as ds:
            self.base_hrrr_channels = list(ds.channel.values)
            self.hrrr_lat = ds.latitude
            self.hrrr_lon = ds.longitude
        
        if self.hrrr_window is None:
            self.hrrr_window = ((0, self.hrrr_lat.shape[0]), (0, self.hrrr_lat.shape[1]))
        
        self.n_samples_total = self.compute_total_samples()

    def __len__(self):
        return len(self.valid_samples)

    def crop_to_fit(self, x):
        '''
        Crop HRRR to get nicer dims
        '''
        ((y0, y1), (x0, x1)) = self._get_crop_box()
        return x[..., y0:y1, x0:x1]

    def to_datetime(self, date):
        timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                     / np.timedelta64(1, 's'))
        return datetime.utcfromtimestamp(timestamp)

    def compute_total_samples(self):
        '''
        Loop through all years and count the total number of samples
        '''
        first_year = min(self.years)
        last_year = max(self.years)
        # if not self.use_all and first_year <= 2018:
        if not self.all_times and first_year <= 2018:
            first_sample = datetime(year=2018, month=8, day=1, hour=1, minute=0, second=0) #marks transition of hrrr model version
        else:
            first_sample = datetime(year=first_year, month=1, day=1, hour=0, minute=0, second=0)
        last_sample = datetime(year=last_year, month=12, day=31, hour=23, minute=0, second=0)

        logging.info("First sample is {}".format(first_sample)) 
        logging.info("Last sample is {}".format(last_sample))

        all_datetimes = time_range(first_sample, last_sample, step=timedelta(hours=1), inclusive=True)
        all_datetimes = set(dt for dt in all_datetimes if dt.year in self.years)

        samples_file = os.path.join(self.location, 'missing_samples.npy')
        if self.s3:
            samples_file = self.s3.open(samples_file)
        missing_samples = np.load(samples_file, allow_pickle=True)
        
        # add two unreliable samples
        m1 = datetime(2018, 3, 11, 3, 0)
        m2 = datetime(2018, 3, 11, 4, 0)
        missing_samples = np.append(missing_samples, [m1,m2])

        missing_samples = set(missing_samples) #hash for faster lookup
        self.valid_samples = sorted(all_datetimes.difference(missing_samples)) # exclude missing samples
        logging.info('Total datetimes in training set are {} of which {} are valid'.format(len(all_datetimes), len(self.valid_samples)))

        if self.shard: # use only part of dataset in each training process
            dist_manager = DistributedManager()
            self.valid_samples = np.array_split(self.valid_samples, dist_manager.world_size)[dist_manager.rank]

        return len(self.valid_samples)

    def normalize_input(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.means_era5
            x /= self.stds_era5
        return x

    def denormalize_input(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x *= self.stds_era5
            x += self.means_era5
        return x

    def _get_era5(self, ts, lat, lon):
        '''
        Retrieve ERA5 samples from zarr files
        '''
        era5_handle = self._get_ds_handles(self.ds_era5, self.era5_paths, ts)
        era5_field = era5_handle.sel(time=ts, channel=self.kept_era5_channel_names).interp(latitude=lat, longitude=lon).data.values
        era5_field = self.normalize_input(era5_field)

        return era5_field

    def normalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x -= self.means_hrrr
            x /= self.stds_hrrr
        return x

    def denormalize_output(self, x):
        x = x.astype(np.float32)
        if self.normalize:
            x *= self.stds_hrrr
            x += self.means_hrrr
        return x

    def _get_hrrr(self, ts, crop_box):
        '''
        Retrieve HRRR samples from zarr files
        '''
        
        hrrr_handle = self._get_ds_handles(self.ds_hrrr, self.hrrr_paths, ts, mask_and_scale=False)
        ds_channel_names = list(np.array(hrrr_handle.channel))
        ((y0, y1), (x0, x1)) = crop_box

        hrrr_field = hrrr_handle.sel(time=ts, channel=self.kept_hrrr_channel_names).HRRR[:, y0:y1, x0:x1].values
        hrrr_field = self.normalize_output(hrrr_field)
        return hrrr_field

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        ((y_start, y_end), (x_start, x_end)) = self.hrrr_window
        return (y_end - y_start, x_end - x_start)

    def _get_crop_box(self):
        if self.sample_shape == None:
            return self.hrrr_window

        ((y_start, y_end), (x_start, x_end)) = self.hrrr_window

        y0 = np.random.randint(y_start, y_end - self.sample_shape[0] + 1)
        y1 = y0 + self.sample_shape[0]
        x0 = np.random.randint(x_start, x_end - self.sample_shape[1] + 1)
        x1 = x0 + self.sample_shape[1]
        return ((y0, y1), (x0, x1))

    def __getitem__(self, global_idx):
        '''
        Return data as a dict (so we can potentially add extras, metadata, etc if desired
        '''
        torch.cuda.nvtx.range_push("hrrr_dataloader:get")
        if self.overfit:
            global_idx = 42
        time_index = self._global_idx_to_datetime(global_idx)
        ((y0, y1), (x0, x1)) = crop_box = self._get_crop_box()
        lon = self.hrrr_lon[y0:y1, x0:x1]
        lat = self.hrrr_lat[y0:y1, x0:x1]
        era5_sample = self._get_era5(time_index, lon=lon, lat=lat)
        if self.ds_factor > 1:
            era5_sample = self._create_lowres_(era5_sample, factor=self.ds_factor)
        hrrr_sample = self._get_hrrr(time_index, crop_box=crop_box)

        torch.cuda.nvtx.range_pop()
        # expand = True
        # if (expand):
        #     # 7.5 million pixels
        #     hrrr_sample = np.pad(hrrr_sample,((0,0),(0,self.y_expand),(0,self.x_expand)))
        #     era5_sample = np.pad(era5_sample,((0,0),(0,self.y_expand),(0,self.x_expand)))
        #     # # 9.2 million pixels
        #     # hrrr_sample = np.pad(hrrr_sample,((0,0),(0,1504),(0,1792)))
        #     # era5_sample = np.pad(era5_sample,((0,0),(0,1504),(0,1792)))
        #     # # 10.5 million pixels
        #     # hrrr_sample = np.pad(hrrr_sample,((0,0),(0,1864),(0,1792)))
        #     # era5_sample = np.pad(era5_sample,((0,0),(0,1864),(0,1792)))
        #print(f'sizes {hrrr_sample.shape} {era5_sample.shape}')
        return hrrr_sample, era5_sample, global_idx

    def _global_idx_to_datetime(self, global_idx):
        '''
        Parse a global sample index and return the input/target timstamps as datetimes
        '''
        return self.valid_samples[global_idx]

    def _get_ds_handles(self, handles, paths, ts, mask_and_scale=True):
        '''
        Return handles for the appropriate year
        '''
        year = ts.year
        if year in handles:
            ds_handle = handles[year]
        else:
            ds_handle = xr.open_zarr(paths[year], consolidated=True, mask_and_scale=mask_and_scale)
            handles[year] = ds_handle
        return ds_handle

    @staticmethod
    def _create_lowres_(x, factor=4):
        # downsample the high res imag
        x = x.transpose(1, 2, 0)
        x = x[::factor, ::factor, :]  # 8x8x3  #subsample
        # upsample with bicubic interpolation to bring the image to the nominal size
        x = cv2.resize(
            x, (x.shape[1] * factor, x.shape[0] * factor), interpolation=cv2.INTER_CUBIC
        )  # 32x32x3
        x = x.transpose(2, 0, 1)  # 3x32x32
        return x

    def latitude(self):
        return self.hrrr_lat if self.train else self.crop_to_fit(self.hrrr_lat)

    def longitude(self):
        return self.hrrr_lon if self.train else self.crop_to_fit(self.hrrr_lon)

    def time(self):
        return self.valid_samples

    def input_channels(self):
        return [ChannelMetadata(name=n) for n in self._get_era5_channel_names()]

    def output_channels(self):
        return [ChannelMetadata(name=n) for n in self._get_hrrr_channel_names()]

    def info(self):
        return {
            "input_normalization": (self.means_era5.squeeze(), self.stds_era5.squeeze()),
            "target_normalization": (self.means_hrrr.squeeze(), self.stds_hrrr.squeeze())
        }
