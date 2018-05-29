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
### For applying in a 3D ROMS model

import numpy as np
import logging
import pyproj
from scipy.interpolate import interp1d
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from opendrift.models.opendrift3D import OpenDrift3DSimulation


def vertical_module(self):
    """Give to particles a  vertical speed according to buoyancy or settlement velocity at PL
    
    Buoyancy is expressed as terminal velocity, which is the
    steady-state vertical velocity due to positive or negative
    buoyant behaviour. It is usually a function of particle density,
    diameter, and shape.
    The formulation of this scheme is copied from LADIM (IMR) and from OpenDrift Scripts.
    """
    
    if (self.get_config('processes:verticalmodule') is False):
        logging.debug('Vertical module deactivated.')
        return

    dt = self.time_step.total_seconds()

    # minimum height/maximum depth for each particle
    Zmin = -1.*self.environment.sea_floor_depth_below_sea_level


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

    #remember which particles belong to the exact surface
    if self.get_config('processes:Buoyancy') is True:
        # update terminal velocity according to environmental variables
        if self.get_config('turbulentmixing:TSprofiles') is True:
            self.update_terminal_velocity(Tprofiles=Tprofiles,
                                            Sprofiles=Sprofiles,
                                            z_index=z_index)
        else:
            # this is faster, but ignores density gradients in
            # water column for the inner loop
            self.update_terminal_velocity()
        w = self.elements.terminal_velocity
        logging.debug('Terminal Velocity is ' + str(np.mean(self.elements.terminal_velocity)))
    else:
        w = 0

    if self.get_config('processes:Settlement') is True: # Personal added line to make particles going in the vertical layer, now, just a constant velocity activated at PostLarvae stages.
        w += self.elements.particles_velocity
        logging.debug('Larval Velocity is ' + str(np.mean(self.elements.particles_velocity)))


    self.elements.z = self.elements.z + dt*w # move according to buoyancy and/or settlement velocity.

    surface = np.where(self.elements.z > 0)
    # put the particles that belong to the surface slick (if present) back to the surface
    self.elements.z[surface] = 0.

    #avoid that elements are below bottom
    bottom = np.where(self.elements.z < Zmin)

    if len(bottom[0]) > 0:
        self.elements.z[bottom] = np.round(Zmin[bottom]) + 0.5



def vertical_mixing(self):
    """Mix particles vertically according to eddy diffusivity and buoyancy
            Buoyancy is expressed as terminal velocity, which is the
            steady-state vertical velocity due to positive or negative
            buoyant behaviour. It is usually a function of particle density,
            diameter, and shape.

            Vertical particle displacemend du to turbulent mixing is
            calculated using a random walk scheme" (Visser et al. 1996)
    """

    if self.get_config('processes:turbulentmixing') is False:
        logging.debug('Turbulent mixing deactivated')
        return

    self.timer_start('main loop:updating elements:vertical mixing')
    from opendrift.models import eddydiffusivity

    dt_mix = self.get_config('turbulentmixing:timestep')

    # minimum height/maximum depth for each particle
    Zmin = -1.*self.environment.sea_floor_depth_below_sea_level

    # Eventual model specific preparions
    self.prepare_vertical_mixing()

    # get profile of eddy diffusivity
    # get vertical eddy diffusivity from environment or specific model
    if (self.get_config('turbulentmixing:diffusivitymodel') ==
            'environment'):
        if 'ocean_vertical_diffusivity' in self.environment_profiles:
            Kprofiles = self.environment_profiles[
                'ocean_vertical_diffusivity']
            logging.debug('Using diffusivity from ocean model')
        else:
            # NB: using constant diffusivity, and value from first
            # element only - this should be checked/improved!
            Kprofiles = \
                self.environment.ocean_vertical_diffusivity[0] * \
                np.ones((len(self.environment_profiles['z']),
                            self.num_elements_active()))
            logging.debug('Using constant diffusivity')
    else:
        logging.debug('Using functional expression for diffusivity')
        Kprofiles = getattr(
            eddydiffusivity,
            self.get_config('turbulentmixing:diffusivitymodel'))(self)

    logging.debug('Diffiusivities are in range %s to %s' %
                    (Kprofiles.min(), Kprofiles.max()))

    # get profiles of salinity and temperature
    # (to save interpolation time in the inner loop)
    if (self.get_config('turbulentmixing:TSprofiles') is True
        and 'sea_water_salinity' in self.required_variables):
        Sprofiles = self.environment_profiles['sea_water_salinity']
        Tprofiles = \
            self.environment_profiles['sea_water_temperature']
        if ('sea_water_salinity' in self.fallback_values and
            Sprofiles.min() == Sprofiles.max()):
            logging.debug('Salinity and temperature are fallback'
                            'values, skipping TSprofile')
            Sprofiles = None
            Tprofiles = None
        else:
            logging.debug('Using TSprofiles for vertical mixing')
    else:
        logging.debug('TSprofiles deactivated for vertical mixing')
        Sprofiles = None
        Tprofiles = None

    # prepare vertical interpolation coordinates
    #z_i = range(Kprofiles.shape[0])
    z_i = range(self.environment_profiles['z'].shape[0])
    #print len(self.environment_profiles['z']), len(z_i)
    z_index = interp1d(-self.environment_profiles['z'],
                        z_i, bounds_error=False,
                        fill_value=(0,len(z_i)-1))  # Extrapolation

    # internal loop for fast time step of vertical mixing model
    # random walk needs faster time step compared
    # to horizontal advection
    logging.debug('Vertical mixing module:')
    ntimes_mix = np.abs(int(self.time_step.total_seconds()/dt_mix))
    logging.debug('Turbulent diffusion with random walk '
                    'scheme using ' + str(ntimes_mix) +
                    ' fast time steps of dt=' + str(dt_mix) + 's')

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

        w = self.elements.terminal_velocity

        # diffusivity K at depth z+dz
        dz = 1e-3
        zi = z_index(-self.elements.z+0.5*dz)
        upper = np.maximum(np.floor(zi).astype(np.int), 0)
        lower = np.minimum(upper+1, Kprofiles.shape[0]-1)
        weight_upper = 1 - (zi - upper)
        weight_upper[np.isnan(weight_upper)] = 1
        K1 = Kprofiles[upper, range(Kprofiles.shape[1])] * \
            weight_upper + \
            Kprofiles[lower, range(Kprofiles.shape[1])] * \
            (1-weight_upper)

        # diffusivity K at depth z-dz
        zi = z_index(-self.elements.z-0.5*dz)
        upper = np.maximum(np.floor(zi).astype(np.int), 0)
        lower = np.minimum(upper+1, Kprofiles.shape[0]-1)
        weight_upper = 1 - (zi - upper)
        weight_upper[np.isnan(weight_upper)] = 1
        K2 = Kprofiles[upper, range(Kprofiles.shape[1])] * \
            weight_upper + \
            Kprofiles[lower, range(Kprofiles.shape[1])] * \
            (1-weight_upper)

        # diffusivity gradient
        dKdz = (K1 - K2) / dz

        # K at depth z+dKdz*dt/2
        zi = z_index(-(self.elements.z+dKdz*dt_mix/2))
        upper = np.maximum(np.floor(zi).astype(np.int), 0)
        lower = np.minimum(upper+1, Kprofiles.shape[0]-1)
        weight_upper = 1 - (zi - upper)
        weight_upper[np.isnan(weight_upper)] = 1
        K3 = Kprofiles[upper, range(Kprofiles.shape[1])] * \
            weight_upper + \
            Kprofiles[lower, range(Kprofiles.shape[1])] * \
            (1-weight_upper)


        # Visser et al. 1996 random walk mixing
        # requires an inner loop time step dt such that
        # dt << (d2K/dz2)^-1, e.g. typically dt << 15min
        R = 2*np.random.random(self.num_elements_active()) - 1
        r = 1.0/3

        depth_t=self.elements.z # depth before the mixing process ADDED
        Condition_MLD=np.where(depth_t < self.environment.surface_boundary_layer)
        # new position  =  old position   - up_K_flux   + random walk
        if self.get_config('turbulentmixing:moduleturb')==3:
            self.elements.z[Condition_MLD]=self.elements.z[Condition_MLD]- dKdz[Condition_MLD]*dt_mix + R[Condition_MLD]*np.sqrt(( K3[Condition_MLD]*dt_mix*2/r))
        else:
            self.elements.z = self.elements.z - dKdz*dt_mix + R*np.sqrt(( K3*dt_mix*2/r))
 
        # Reflect from surface
        reflect = np.where(self.elements.z >= 0)
        if len(reflect[0]) > 0:
            self.elements.z[reflect] = -self.elements.z[reflect]

        # reflect elements going below seafloor
        bottom = np.where(self.elements.z < Zmin)
        if len(bottom[0]) > 0:
            logging.debug('%s elements penetrated seafloor, lifting up' % len(bottom[0]))
            self.elements.z[bottom] = 2*Zmin[bottom] - self.elements.z[bottom]

        if self.get_config('processes:Buoyancy') is True: #### added
            # advect due to buoyancy
            self.elements.z = self.elements.z + w*dt_mix

            # put the particles that belonged to the surface slick (if present) back to the surface
        self.elements.z[surface] = 0.

            # formation of slick and wave mixing for surfaced particles if implemented for this class
        self.surface_stick()
        self.surface_wave_mixing(dt_mix)

            # let particles stick to bottom 
        bottom = np.where(self.elements.z < Zmin)
        if len(bottom[0]) > 0:
            logging.debug('%s elements reached seafloor, set to bottom' % len(bottom[0]))
            self.elements.z[bottom] = Zmin[bottom]
 
    self.timer_end('main loop:updating elements:vertical mixing')




def horizontal_mixing(self,method):
    """Mix particles Horizontally according to eddy diffusivity
       
        Horizontal particle displacemend du to turbulent mixing is
        calculated using the  "random walk scheme"  with a random buffer taken in a Gaussian distribution of mean =0 and standard deviation 1.
    """

    if self.get_config('processes:horiz_turbulent') is False:
        logging.debug('Horizontal diffusion deactivated')
        return
    depth_t=self.elements.z # depth before the mixing process

    MLD=self.environment.surface_boundary_layer
    MLD=np.array(MLD)
    Condition_MLD=depth_t > MLD.astype(float)

    if (method=='output'):
        # get horizontal eddy diffusivity from environment or specific model
        if (self.get_config('turbulentmixing:diffusivityhorizontal')=='environment'):
            if 'ocean_horizontal_diffusivity' in self.environment_profiles:
                Kh = self.environment_profiles['ocean_horizontal_diffusivity']
                logging.debug('Use diffusivity from ocean model')
                logging.debug('Diffiusivities are in range %s to %s.' %
                              (Kh.min(), Kh.max()))
            else:
                # NB: using constant diffusivity, and value from first
                # element only - this should be checked/improved!
                Kh = self.environment.ocean_horizontal_diffusivity[0] * \
                        np.ones((len(self.environment_profiles['z']),
                                 self.num_elements_active()))
                logging.debug('Use constant diffusivity %s' %(Kh))
        else:
            logging.debug('Use given diffusivity')
            Kh = self.fallback_values['ocean_horizontal_diffusivity']

        if self.get_config('turbulentmixing:moduleturb')==3:
            sigma_u=np.zeros(self.num_elements_active())
            sigma_v=np.zeros(self.num_elements_active())
            np.random.seed()
            sigma_u[Condition_MLD]=np.random.normal(0, 1,sum(Condition_MLD))*np.sqrt(2.*Kh/self.time_step.total_seconds())
            np.random.seed()
            sigma_v[Condition_MLD]=np.random.normal(0, 1,sum(Condition_MLD))*np.sqrt(2.*Kh/self.time_step.total_seconds())
        else:
            np.random.seed()
            sigma_u = np.random.normal(0, 1,self.num_elements_active())*np.sqrt(2.*Kh/self.time_step.total_seconds()) ### According to Okubo, 1971
            np.random.seed()
            sigma_v = np.random.normal(0, 1,self.num_elements_active())*np.sqrt(2.*Kh/self.time_step.total_seconds()) ### According to Guizien et al. 2006

    if (method=='okubo'):
        # get horizontal length squares from environment or specific model
        # Metho Okubo is extracted from Okubo (1971)
        # It's computing Coefficient of apparent diffusivity Ka from the length scale
        
        if (self.get_config('turbulentmixing:diffusivityhorizontal')=='environment'):
            if 'turbulent_generic_length_scale' in self.environment_profiles:
                gls = self.environment_profiles['turbulent_generic_length_scale']
                logging.debug('Use gls from ocean model')
                logging.debug('GLS are in range %s to %s.' %
                                  (gls.min(), gls.max()))
            else:
                # NB: using constant diffusivity, and value from first
                # element only - this should be checked/improved!
                gls = self.environment.turbulent_generic_length_scale[0] * \
                    np.ones((len(self.environment_profiles['z']),
                                 self.num_elements_active()))
                logging.debug('Use constant GLS %s' %(gls))
        else:
            logging.debug('Use given GLS')
            gls = self.fallback_values['turbulent_generic_length_scale']

        Kh= 10.0 #0.0103*(2000**1.15)*(10**-4) # Equation from Okubo is giving Kh in cm/s2 --> conversion in m2/s by *

        if self.get_config('turbulentmixing:moduleturb')==3:
            sigma_u=np.zeros(self.num_elements_active())
            sigma_v=np.zeros(self.num_elements_active())
            np.random.seed()
            sigma_u[Condition_MLD]=np.random.uniform(0, 1,sum(Condition_MLD))*np.sqrt(2.*Kh/np.abs(self.time_step.total_seconds()))
            np.random.seed()
            sigma_v[Condition_MLD]=np.random.uniform(0, 1,sum(Condition_MLD))*np.sqrt(2.*Kh/np.abs(self.time_step.total_seconds()))
        else:
            np.random.seed()
            sigma_u = np.random.uniform(0, 1,self.num_elements_active())*np.sqrt(Kh/np.abs(self.time_step.total_seconds())) ### According to Okubo, 1971
            np.random.seed()
            sigma_v = np.random.uniform(0, 1,self.num_elements_active())*np.sqrt(Kh/np.abs(self.time_step.total_seconds())) ### According to Guizien et al. 2006


    if (method=='guizien'):
        # Method Guizien is extracted from Guizien et al. ()
        # It's supposing that sigma_u and sigma_v are extracted from a Gaussian Distribution (0,sqrt(TKE*2/3)
        # get TKE from environment or specific model
        if (self.get_config('turbulentmixing:diffusivityhorizontal')=='environment'):
            if 'turbulent_kinetic_energy' in self.environment_profiles:
                TKE = self.environment_profiles['turbulent_kinetic_energy']
                logging.debug('Use TKE from ocean model')
                logging.debug('Length scales are in range %s to %s.' %
                                (TKE.min(), TKE.max()))
            else:
                # NB: using constant diffusivity, and value from first
                # element only - this should be checked/improved!
                TKE = self.environment.turbulent_kinetic_energy[0] * \
                    np.ones((len(self.environment_profiles['z']),
                                 self.num_elements_active()))
                logging.debug('Use constant TKE %s' %(TKE))
        else:
            logging.debug('Use given TKE')
            TKE = self.fallback_values['turbulent_kinetic_energy']

        if self.get_config('turbulentmixing:moduleturb')==3:
            sigma_u=np.zeros(self.num_elements_active())
            sigma_v=np.zeros(self.num_elements_active())
            np.random.seed()
            sigma_u[Condition_MLD]=np.random.normal(0, np.sqrt(TKE*2/3),sum(Condition_MLD))*np.abs(self.time_step.total_seconds())
            np.random.seed()
            sigma_v[Condition_MLD]=np.random.normal(0, np.sqrt(TKE*2/3),sum(Condition_MLD))*np.abs(self.time_step.total_seconds())
        else:
            np.random.seed()
            sigma_u = np.random.normal(0, np.sqrt(TKE*2/3),self.num_elements_active())*np.abs(self.time_step.total_seconds())
            np.random.seed()
            sigma_v = np.random.normal(0, np.sqrt(TKE*2/3),self.num_elements_active())*np.abs(self.time_step.total_seconds())

#self.update_positions(sigma_u,sigma_v)

    self.environment.x_sea_water_velocity += sigma_u
    self.environment.x_sea_water_velocity += sigma_v
