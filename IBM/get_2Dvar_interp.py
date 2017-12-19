# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway
###################################################

# Function to interpolate the variable sea_floor_depth_below_sea_level saved in the 'o' IBM to the coordinates of the particles.
# Adapted from get_variables_interpolated in basereader.py
import sys
import logging
from bisect import bisect_left
from datetime import timedelta, datetime
from scipy.interpolate import interp2d

from opendrift.readers.basereader import BaseReader, vector_pairs_xy
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
import numpy as np
from netCDF4 import MFDataset

from opendrift.readers.interpolation import ReaderBlock


def get_2DBathy_interp(self, variables, time=None,
                                lon=None, lat=None,
                                rotate_to_proj=None):
    ### Set the z-axes information to None
    z=None
    block=False
    profiles=None
    profiles_depth=None
    
    # Raise error if time not not within coverage of reader
    if not self.covers_time(time):
        raise ValueError('%s is outside time coverage (%s - %s) of %s' %
                            (time, self.start_time, self.end_time, self.name))

    # Check which particles are covered (indep of time)
    ind_covered = self.covers_positions(lon, lat)
    if len(ind_covered) == 0:
        raise ValueError(('All %s particles (%.2f-%.2fE, %.2f-%.2fN) ' +
                            'are outside domain of %s (%s)') %
                            (len(lon), lon.min(), lon.max(), lat.min(),
                            lat.max(), self.name, self.coverage_string()))

    # Find reader time_before/time_after
    time_nearest, time_before, time_after, i1, i2, i3 = \
        self.nearest_time(time)
    logging.debug('Reader time:\n\t\t%s (before)\n\t\t%s (after)' %
                    (time_before, time_after))
    if time == time_before:
        time_after = None

    reader_x, reader_y = self.lonlat2xy(lon[ind_covered],lat[ind_covered])
            
    if block is False or self.return_block is False:
        # Analytical reader, continous in space and time
        env_before = self._get_variables(variables, profiles,
                                            profiles_depth,
                                            time,
                                            #time_before,
                                            reader_x, reader_y, z,
                                            block=block)
        logging.debug('Fetched env-before')

    else:
        # Swap before- and after-blocks if matching times
        if str(variables) in self.var_block_before:
            block_before_time = self.var_block_before[
                str(variables)].time
            if str(variables) in self.var_block_after:
                block_after_time = self.var_block_after[
                    str(variables)].time
                if block_before_time != time_before:
                    if block_after_time == time_before:
                        self.var_block_before[str(variables)] = \
                            self.var_block_after[str(variables)]
                if block_after_time != time_after:
                    if block_before_time == time_before:
                        self.var_block_after[str(variables)] = \
                            self.var_block_before[str(variables)]
        # Fetch data, if no buffer is available
        if (not str(variables) in self.var_block_before) or \
                (self.var_block_before[str(variables)].time !=
                    time_before):
            reader_data_dict = \
                self._get_variables(variables, profiles,
                                    profiles_depth, time_before,
                                    reader_x, reader_y, z,
                                    block=block)
            self.var_block_before[str(variables)] = \
                ReaderBlock(reader_data_dict,
                            interpolation_horizontal=self.interpolation)

        if not str(variables) in self.var_block_after or \
                self.var_block_after[str(variables)].time != time_after:
            if time_after is None:
                self.var_block_after[str(variables)] = \
                    self.var_block_before[str(variables)]
            else:
                reader_data_dict = \
                    self._get_variables(variables, profiles,
                                        profiles_depth, time_after,
                                        reader_x, reader_y, z,
                                        block=block)
                self.var_block_after[str(variables)] = \
                    ReaderBlock(
                        reader_data_dict,
                        interpolation_horizontal=self.interpolation)

        if self.var_block_before[str(variables)].covers_positions(
            reader_x, reader_y) is False or \
            self.var_block_after[str(variables)].covers_positions(
                reader_x, reader_y) is False:
            logging.warning('Data block from %s not large enough to '
                            'cover element positions within timestep. '
                            'Buffer size (%s) must be increased.' %
                            (self.name, str(self.buffer)))

        ############################################################
        # Interpolate before/after blocks onto particles in space
        ############################################################
        logging.debug('Interpolating before (%s) in space  (%s)' %
                        (self.var_block_before[str(variables)].time,
                        self.interpolation))
        env_before, env_profiles_before = self.var_block_before[
            str(variables)].interpolate(
                reader_x, reader_y, z, variables,
                profiles, profiles_depth)

    #######################
    # Time interpolation
    #######################
    env_profiles = None
    logging.debug('No time interpolation needed.')
    env = env_before

    ####################
    # Rotate vectors
    ####################
    if rotate_to_proj is not None:
        if (rotate_to_proj.srs == self.proj.srs) or (
            rotate_to_proj.is_latlong() is True and
            self.proj.is_latlong() is True):
            logging.debug('Reader SRS is the same as calculation SRS - '
                            'rotation of vectors is not needed.')
        else:
            vector_pairs = []
            for var in variables:
                for vector_pair in vector_pairs_xy:
                    if var in vector_pair:
                        counterpart = list(set(vector_pair) -
                                            set([var]))[0]
                        if counterpart in variables:
                            vector_pairs.append(vector_pair)
                        else:
                            sys.exit('Missing component of vector pair:' +
                                        counterpart)
                # Extract unique vector pairs
            vector_pairs = [list(x) for x in set(tuple(x)
                            for x in vector_pairs)]

            if len(vector_pairs) > 0:
                for vector_pair in vector_pairs:
                    env[vector_pair[0]], env[vector_pair[1]] = \
                        self.rotate_vectors(reader_x, reader_y,
                                            env[vector_pair[0]],
                                            env[vector_pair[1]],
                                            self.proj, rotate_to_proj)
                    if profiles is not None and vector_pair[0] in profiles:
                        sys.exit('Rotating profiles of vectors '
                                    'is not yet implemented')

        # Masking non-covered pixels
    if len(ind_covered) != len(lon):
        logging.debug('Masking %i elements outside coverage' %
                        (len(lon)-len(ind_covered)))
        for var in variables:
            tmp = np.nan*np.ones(lon.shape)
            tmp[ind_covered] = env[var].copy()
            env[var] = np.ma.masked_invalid(tmp)
            # Filling also fin missing columns
            # for env_profiles outside coverage
            if env_profiles is not None and var in env_profiles.keys():
                tmp = np.nan*np.ones((env_profiles[var].shape[0],
                                        len(lon)))
                tmp[:, ind_covered] = env_profiles[var].copy()
                env_profiles[var] = np.ma.masked_invalid(tmp)

    Bathy= - env[variables]


    return Bathy, env