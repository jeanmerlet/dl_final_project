#!/usr/bin/env python

import os
import numpy as np
import xarray as xr
import csv
import re
import random
import warnings
import sys

class DataReader:
    def __init__(self, verbose=False):
        self.verbose = verbose
        # these are layers we'll read from the netcdf files
        self.layers = ['def', 'pdsi', 'prcptn',
                       'soil', 'swe', 'srad',
                       'vap', 'windspeed' ]
        self.layers = ['prcptn', 'vap']
        # these are layers we'll add on ourselves
        self.extra_layers = ['land']
        self.extra_layers = []

        # xarray complains about our input files - suppress these warnings
        #  unless we're in verbose mode
        if not self.verbose:
            try:
                warnings.filterwarnings('ignore',
                                        category=xr.coding.variables.SerializationWarning)
            except AttributeError:
                pass

    def num_input_layers(self):
        return len(self.layers) + len(self.extra_layers)

    def num_output_layers(self):
        return len(self.layers)

    def scan_input_dir(self, directory, year_min=None, year_max=None, years_only=None):
        '''
        Takes in a directory, determines number of years that are valid with the input data and the directory
        '''
        years = []
        scan_dir = os.path.join(directory, self.layers[0] + '_terra')
        for f in os.listdir(scan_dir):
            m = re.search(r'_(\d+).nc$', f)
            if m:
                y = int(m.group(1))
                # filter if requested
                if year_min and (y < year_min):
                    continue
                if year_max and (y > year_max):
                    continue
                if years_only and (y not in years_only):
                    continue
                years.append(y)
        if self.verbose:
            print('Scan found {} suitable years: {}'.format(len(years),
                                                            ', '.join(map(str, years))))
        assert len(years) > 0
        return years
    
    def validate_years(self, directory, years, lat_points, lon_points):
        '''
        creates layer data and years, ensures we can load and that the shape is correct
        '''
        self.layer_data = {}
        self.valid_years = []
        total_size = 0
        for y in years:
            layers = {}
            for l in self.layers:
                # some of the netcdf files use slightly different names
                relabel = { 'pdsi': 'PDSI',
                            'prcptn': 'ppt',
                            'windspeed': 'ws' }
                nc_label = relabel.get(l, l)
                layer_file = os.path.join(directory, l + '_terra', f'TerraClimate_{nc_label}_{y}.nc')
                try:
                    xarray_data = xr.open_dataset(layer_file)
                    data = xarray_data[nc_label]
                    # sanity-check the shape
                    s = data.shape
                    if s != (12, lat_points, lon_points):
                        raise ValueError(f'wrong data shape: {s}')
                    layers[l] = data
                    total_size += sys.getsizeof(data)
                except Exception as e:
                    print(f'FAILED to read "{layer_file}": {e}')
                    print(f'{y} skipped')
                    layers = None
                    break

            if layers:
                self.layer_data[y] = layers
                self.valid_years.append(y)
        self.valid_years.sort()
        print(f'valid years: {self.valid_years}')
        print(f'total mem size: {total_size}')
        #print(gs.get_size(self.layer_data))

    def get_rectangular_indices(self, subregion):
        # returns the coordinate indices to the given lat and lon *integer* ranges
        lat_range, lon_range = subregion
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range
        lat_min_idx = 2160 - (lat_max * 24)
        lat_max_idx = 2160 - (lat_min * 24)
        lon_min_idx = -(4320 - (lon_min * 24))
        lon_max_idx = -(4320 - (lon_max * 24))
        return lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx

    def get_single_point_indices(self, point):
        # returns the closest valid coordinate indices to the given lat and lon values
        lat, lon = point
        lat_idx = np.abs(np.linspace(90, -90, 4320) - lat).argmin()
        lon_idx = np.abs(np.linspace(-180, 180, 8640) - lon).argmin()
        return lat_idx, lon_idx

    def apply_idx_restriction_to_xy_coords(self, lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx):
        # sets any values outside the lat and lon indices in the is_land coordinate set to False
        self.is_land[:lat_min_idx-1, :] = False
        self.is_land[lat_max_idx:, :] = False
        self.is_land[:, :lon_min_idx-1] = False
        self.is_land[:, lon_max_idx:] = False

    def compute_land_file(self, lat_points, subregion, point):
        print('Computing land locations...', end='', flush=True)
        # any layer should do
        l = self.layer_data[self.valid_years[0]][self.layers[0]]
        self.is_land = np.logical_not(np.isnan(l.data[0,:,:]))
        # trim off polar regions (60+ latitude)
        self.is_land[0:(lat_points // 6), :] = False
        self.is_land[(5 * lat_points // 6):, :] = False
        if subregion is not None:
            lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx = self.get_rectangular_indices(subregion)
            self.apply_idx_restriction_to_xy_coords(lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx)
        elif point is not None:
            lat_idx, lon_idx = self.get_single_point_indices(point)
            self.apply_idx_restriction_to_xy_coords(lat_idx, lat_idx, lon_idx, lon_idx)
        self.land_xys = list(zip(*self.is_land.nonzero()))
        #self.land_xys = list(zip(*([self.is_land.nonzero()[0][0]], [self.is_land.nonzero()[1][0]])))
        print(f'{len(self.land_xys)} points found on land')

    def save_land(self, land_xy_file):
        try:
            np.save(land_xy_file, self.land_xys)
            if self.verbose:
                print('saved land xy data to "{}"'.format(land_xy_file))
        except Exception as e:
            print('FAILED to write land xy data to "{}": {}'.format(land_xy_file, e))

    def load_land_file(self, land_xy_file, lat_points, lon_points):
        try:
            self.land_xys = np.load(land_xy_file)
            if self.verbose:
                print('{} points loaded from file'.format(len(self.land_xys)))
            self.is_land = np.full((lat_points, lon_points), False)
            self.is_land[self.land_xys[:, 0], self.land_xys[:, 1]] = True
            self.land_xys = self.land_xys.T
            self.land_xys = list(zip(self.land_xys[0], self.land_xys[1]))
        except FileNotFoundError:
            pass

    def scan_input_data(self, data_root, land_xy_file, subregion=None, point=None,
                        year_min=None, year_max=None, years_only=None):
        lat_points, lon_points = 4320, 8640
        years = self.scan_input_dir(data_root, year_min, year_max, years_only)
        self.validate_years(data_root, years, lat_points, lon_points)
        # read or generate list of land xy locations
        self.land_xys = None
        if land_xy_file: self.load_land_file(land_xy_file, lat_points, lon_points)
        if self.land_xys is None: #the previous line would have updated self.land_xys if a file existed
            self.compute_land_file(lat_points, subregion, point)
            if land_xy_file: self.save_land(land_xy_file)
        #print(self.land_xys)
        #print(type(self.land_xys))

    def configure_batch(self, batch_size, window_size, dtype):
        self.batch_size = batch_size
        self.window_size = window_size
        self.dtype = dtype

    def rnn_data(self, step):
        '''
        takes in a step size, outputs the input data and the target data 
        input: step size (int)

        output:
        in_data(stepxNxn_layers array where N is the number of possible steps)
        tgt_data(1xN array)
        '''
        start_y = self.valid_years[0] #first year
        end_y = self.valid_years[-1] #last years
        
        in_dat = []
        tgt_dat = []
        # while np.isnan(window_data).any():
        lat, lon = random.choice(self.land_xys)
        layer_data = self.layer_data[start_y][self.layers[0]]
        window_data = layer_data[:, lat, lon]
        
        
        for l in self.layers:
            tostack = []
            totgt = []
            for ref_y in range(start_y, end_y+1):
                layer_data = self.layer_data[ref_y][l]
                window_data = layer_data[:, (lat-self.window_size):(lat+self.window_size+1), (lon-self.window_size):(lon+self.window_size+1)]
                tostack.append(window_data.astype(self.dtype))
                tgt = layer_data[:, lat, lon]
                totgt.append(tgt.astype(self.dtype))
            totaldat = np.concatenate(np.stack(tostack))
            print(totaldat.shape)
            tgt = np.concatenate(np.stack(totgt), axis = None)
            in_layer = []
            tgt_layer = []
            for index in range((len(self.valid_years)*12) - step):
                in_layer.append(totaldat[index:index+step, :, :])
                tgt_layer.append(tgt[index+step])
            in_dat.append(np.stack(in_layer))
            tgt_dat.append(np.stack(tgt_layer))
        in_dat = np.squeeze(in_dat)
        print(in_dat.shape)
        tgt_dat = np.squeeze(tgt_dat)
        if len(self.layers) > 1:
            in_dat = np.moveaxis(in_dat, 0, -1)
            tgt_dat = np.moveaxis(tgt_dat, 0, -1)
        print(in_dat.shape)
        print(tgt_dat.shape)
        #reshape for keras
        in_dat = np.reshape(in_dat, (in_dat.shape[0], in_dat.shape[1], len(self.layers)*(in_dat.shape[2]**2)))
        tgt_dat = np.reshape(tgt_dat, (tgt_dat.shape[0], len(self.layers)))
        return in_dat, tgt_dat

    def next_batch(self, ny):
        # samples in a given batch will always use the same reference year
        # don't allow the last year as we need it for loss
        start_y = random.choice(self.valid_years[:-ny])
        # TODO: add logic to correctly calculate reference year for a RNN
        tgt_y = start_y + ny

        in_data = []
        tgt_data = []

        for b in range(self.batch_size):
            # pick an xy that doesn't fall off the edge (TODO: handle wrapping)
            # check if any of the window is in the ocean (coastlines are not straight)
            #
            # this assumes that valid values for the first layer are valid for all
            window_data = np.nan
            while np.isnan(window_data).any():
                lat, lon = random.choice(self.land_xys)
                layer_data = self.layer_data[start_y][self.layers[0]]
                window_data = layer_data[:, (lat - self.window_size) : (lat + self.window_size + 1),
                                            (lon - self.window_size) : (lon + self.window_size + 1)]

            for ref_y in range(start_y, tgt_y):
                tostack = []
                for l in self.layers:
                    layer_data = self.layer_data[ref_y][l]
                    window_data = layer_data[:, (lat - self.window_size) : (lat + self.window_size + 1),
                                            (lon - self.window_size) : (lon + self.window_size + 1)]
                    tostack.append(window_data.astype(self.dtype))
                # now add on extra layers, if any
                for el in self.extra_layers:
                    if el == 'land':
                        window_data = self.is_land[(lat - self.window_size) : (lat + self.window_size + 1),
                                               (lon - self.window_size) : (lon + self.window_size + 1)]
                        tostack.append(window_data.astype(self.dtype))

                in_data.append(np.stack(tostack))

            # we also need single-location climate data for the target year
            tgt_data.append([ self.layer_data[tgt_y][l][:, lat, lon] for l in self.layers ])

        # stack/numpy-ify everything
        in_data = np.stack(in_data)
        tgt_data = np.array(tgt_data, dtype=self.dtype)

        return in_data, tgt_data
    def climate_data(self):
        '''
        Creates climate data for each point we are interested in
        returns XxLx12 matrix where X is the number of points in land_xy and L is   the number of layers
        '''
        climate_data = []
        for lat, lon in self.land_xy:
            pointclim = []
    
            for l in self.layers:
                tostack = []
                for y in self.valid_years[:-1]:
                    year_dat = self.layer_data[y][l]
                    year_dat = year_dat[:, lat, lon]
                    tostack.append(year_dat)
                clim = np.stack(tostack)
                clim = np.mean(clim, axis = 0)
                pointclim.append(clim)
            pointclim=np.stack(pointclim)
            climate_data.append(pointclim)

        return np.array(climate_data)
