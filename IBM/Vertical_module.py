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

### COPIED FROM OPENDRIFT3DSIMULATION the Vertical_Mixing function

import numpy as np
import logging
from scipy.interpolate import interp1d
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray

def vertical_module(self):
    """Mix particles vertically according to eddy diffusivity and buoyancy
    
    Buoyancy is expressed as terminal velocity, which is the
    steady-state vertical velocity due to positive or negative
    buoyant behaviour. It is usually a function of particle density,
    diameter, and shape.
    
    Vertical particle displacemend du to turbulent mixing is
    calculated using the "binned random walk scheme" (Thygessen and
    Aadlandsvik, 2007).
    The formulation of this scheme is copied from LADIM (IMR).
    """
    
    if (self.get_config('processes:verticalmodule') is False):
        logging.debug('Vertical module deactivated.')
        return

    self.timer_start('main loop:updating elements: Vertical Module')
    from opendrift.models import eddydiffusivity

    dt_mix = self.time_step.total_seconds()

    # minimum height/maximum depth for each particle
    Zmin = -1.*self.environment.sea_floor_depth_below_sea_level

    # place particle in center of bin
    surface = self.elements.z == 0
    self.elements.z[~surface] = np.round(self.elements.z[~surface])

    #avoid that elements are below bottom
    bottom = np.where(self.elements.z < Zmin)
    if len(bottom[0]) > 0:
        self.elements.z[bottom] = np.round(Zmin[bottom]) + 0.5

    # get profiles of salinity and temperature
    # (to save interpolation time in the inner loop)
    if (self.get_config('turbulentmixing:TSprofiles') is True
        and 'sea_water_salinity' in self.required_variables):
        Sprofiles = self.environment_profiles['sea_water_salinity']
        Tprofiles = \
            self.environment_profiles['sea_water_temperature']
        # prepare vertical interpolation coordinates
        z_i = range(Tprofiles.shape[0])
        z_index = interp1d(-self.environment_profiles['z'],
                        z_i, bounds_error=False)
        if ('sea_water_salinity' in self.fallback_values and
            Sprofiles.min() == Sprofiles.max()):
            logging.debug('Salinity and temperature are fallback values, '
                            'skipping TSprofile')
            Sprofiles = None
            Tprofiles = None
    else:
        Sprofiles = None
        Tprofiles = None

    # internal loop for fast time step of vertical mixing model
    # binned random walk needs faster time step compared
    # to horizontal advection
    ntimes_mix = np.abs(int(self.time_step.total_seconds()/dt_mix))
    logging.debug('Vertical mixing module:')
    logging.debug('turbulent diffusion with binned random walk scheme')
    logging.debug('using ' + str(ntimes_mix) + ' fast time steps of dt=' +
                    str(dt_mix) + 's')
    for i in range(0, ntimes_mix):
        #remember which particles belong to the exact surface
        surface = self.elements.z == 0

        # update terminal velocity according to environmental variables
        if self.get_config('turbulentmixing:TSprofiles') is True:
            self.update_terminal_velocity(Tprofiles=Tprofiles,
                                            Sprofiles=Sprofiles,
                                            z_index=z_index)
        else:
            # this is faster, but ignores density gradients in
            # water column for the inner loop
            self.update_terminal_velocity()
            
        logging.debug('Terminal Velocity ' + str(np.mean(self.elements.terminal_velocity)))
        w =  self.elements.terminal_velocity + self.elements.settlement_velocity

        # calculate rise/sink probability dependent on K and w

        self.elements.z = self.elements.z + dt_mix*w # move according to buoyancy and/or settlement velocity.

        # put the particles that belong to the surface slick (if present) back to the surface
        self.elements.z[surface] = 0.

        #avoid that elements are below bottom
        bottom = np.where(self.elements.z < Zmin)
        if len(bottom[0]) > 0:
            self.elements.z[bottom] = np.round(Zmin[bottom]) + 0.5

        # Call surface interaction:
        # reflection at surface or formation of slick and wave mixing if implemented for this class
        self.surface_interaction(dt_mix)