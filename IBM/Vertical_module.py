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

    self.timer_start('main loop:updating elements: Vertical Module')

    dt = self.time_step.total_seconds()

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
        logging.debug('Settlement Velocity is' + str(np.mean(self.elements.particles_velocity)))


    self.elements.z = self.elements.z + dt*w # move according to buoyancy and/or settlement velocity.

    # put the particles that belong to the surface slick (if present) back to the surface
    self.elements.z[surface] = 0.

    #avoid that elements are below bottom
    bottom = np.where(self.elements.z < Zmin)
    if len(bottom[0]) > 0:
        self.elements.z[bottom] = np.round(Zmin[bottom]) + 0.5

    # Call surface interaction:
    # reflection at surface or formation of slick and wave mixing if implemented for this class
    self.surface_interaction(dt)


def vertical_mixing(self):
    """Mix particles vertically according to eddy diffusivity, buoyancy and Settlement Velocity of PL

        Buoyancy is expressed as terminal velocity, which is the
        steady-state vertical velocity due to positive or negative
        buoyant behaviour. It is usually a function of particle density,
        diameter, and shape.

        Vertical particle displacemend du to turbulent mixing is
        calculated using the "binned random walk scheme" (Thygessen and
        Aadlandsvik, 2007).
        The formulation of this scheme is copied from LADIM (IMR).
    """

    if self.get_config('processes:turbulentmixing') is False:
        logging.debug('Turbulent mixing deactivated.')
        return

    self.timer_start('main loop:updating elements:vertical mixing')
    from opendrift.models import eddydiffusivity

    dz = self.get_config('turbulentmixing:verticalresolution')
    dz = np.float32(dz)  # Convert to avoid error for older numpy
    dt_mix = self.get_config('turbulentmixing:timestep')

    # minimum height/maximum depth for each particle
    Zmin = -1.*self.environment.sea_floor_depth_below_sea_level

    # place particle in center of bin
    surface = self.elements.z == 0
    self.elements.z[~surface] = np.round(self.elements.z[~surface]/dz)*dz

    # Prevent elements to go below seafloor
    bottom = np.where(self.elements.z < Zmin)
    if len(bottom[0]) > 0:
        logging.debug('%s elements penetrated seafloor, lifting up' % len(bottom[0]))
        self.elements.z[bottom] = np.round(Zmin[bottom]/dz)*dz + dz/2.

    # Eventual model specific preparions
    self.prepare_vertical_mixing()

    # get profile of eddy diffusivity
    # get vertical eddy diffusivity from environment or specific model
    if (self.get_config('turbulentmixing:diffusivitymodel') ==
            'environment'):
        if 'ocean_vertical_diffusivity' in self.environment_profiles:
            Kprofiles = self.environment_profiles[
                'ocean_vertical_diffusivity']
            logging.debug('use diffusivity from ocean model')
        else:
            # NB: using constant diffusivity, and value from first
            # element only - this should be checked/improved!
            Kprofiles = \
                self.environment.ocean_vertical_diffusivity[0] * \
                np.ones((len(self.environment_profiles['z']),
                            self.num_elements_active()))
            logging.debug('use constant diffusivity')
    else:
        logging.debug('use functional expression for diffusivity')
        Kprofiles = getattr(
            eddydiffusivity,
            self.get_config('turbulentmixing:diffusivitymodel'))(self)

    logging.debug('Diffiusivities are in range %s to %s.' %
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
            logging.debug('Salinity and temperature are fallback values, '
                            'skipping TSprofile')
            Sprofiles = None
            Tprofiles = None
    else:
        Sprofiles = None
        Tprofiles = None

    # prepare vertical interpolation coordinates
    z_i = range(Kprofiles.shape[0])
    z_index = interp1d(-self.environment_profiles['z'],
                        z_i, bounds_error=False,
                        fill_value=(0,len(z_i)-1))  # Extrapolation
    # internal loop for fast time step of vertical mixing model
    # binned random walk needs faster time step compared
    # to horizontal advection
    ntimes_mix = np.abs(int(self.time_step.total_seconds()/dt_mix))
    logging.debug('Vertical mixing module:')
    logging.debug('turbulent diffusion with binned random walk scheme')
    logging.debug('using ' + str(ntimes_mix) + ' fast time steps of dt=' +
                    str(dt_mix) + 's')
    depth_t=self.elements.z # depth before the mixing process
    Condition_MLD=np.where(depth_t < self.environment.surface_boundary_layer)
    for i in range(0, ntimes_mix):
            #remember which particles belong to the exact surface
        surface = self.elements.z == 0
        if self.get_config('processes:Buoyancy') is True:
            # update terminal velocity according to environmental variables if this module is added.
            if self.get_config('turbulentmixing:TSprofiles') is True:
                self.update_terminal_velocity(Tprofiles=Tprofiles,
                                              Sprofiles=Sprofiles,
                                              z_index=z_index)
            else:
                # this is faster, but ignores density gradients in
                # water column for the inner loop
                self.update_terminal_velocity()

            w = self.elements.terminal_velocity
        else:
            w = 0
        if self.get_config('processes:Settlement') is True:
            w += self.elements.particles_velocity

        # diffusivity K at depth z
        zi = z_index(-self.elements.z)
        upper = np.maximum(np.floor(zi).astype(np.int), 0)
        lower = np.minimum(upper+1, Kprofiles.shape[0]-1)
        weight_upper = 1 - (zi - upper)
        weight_upper[np.isnan(weight_upper)] = 1
        K1 = Kprofiles[upper, range(Kprofiles.shape[1])] * \
            weight_upper + \
            Kprofiles[lower, range(Kprofiles.shape[1])] * \
            (1-weight_upper)

        # K at depth z-dz ; gradient of K is required for correct
        # solution with random walk scheme
        zi = z_index(-(self.elements.z-dz))
        upper = np.maximum(np.floor(zi).astype(np.int), 0)
        lower = np.minimum(upper+1, Kprofiles.shape[0]-1)
        weight_upper = 1 - (zi - upper)
        weight_upper[np.isnan(weight_upper)] = 1
        K2 = Kprofiles[upper, range(Kprofiles.shape[1])] * \
            weight_upper + \
            Kprofiles[lower, range(Kprofiles.shape[1])] * \
            (1-weight_upper)

        # calculate rise/sink probability dependent on K and w
        p = dt_mix * (2.0*K1 + dz*w)/(2.0*dz*dz)  # probability to rise
        q = dt_mix * (2.0*K2 - dz*w)/(2.0*dz*dz)  # probability to sink

        # check if probabilities are reasonable or wrong; which can happen if K is very high (K>0.1)
        wrong = p+q > 1.00002
        if wrong.sum() > 0:
            logging.info('WARNING! '+str(wrong.sum())+' elements have p+q>1; you might need a smaller mixing time step')
            # fixing p and q by scaling them to assure p+q<1:
            norm = p+q
            p[wrong] = p[wrong]/norm[wrong]
            q[wrong] = q[wrong]/norm[wrong]

        # use probabilities to mix some particles up or down
        RandKick = np.random.random(self.num_elements_active())
        up = np.where(RandKick < p)
        down = np.where(RandKick > 1.0 - q)
        if self.get_config('turbulentmixing:moduleturb')==3:
            self.elements.z[up and Condition_MLD]=self.elements.z[up and Condition_MLD] + dz
            self.elements.z[down and Condition_MLD] = self.elements.z[down and Condition_MLD] - dz
        else:
            self.elements.z[up] = self.elements.z[up] + dz # move to layer above
            self.elements.z[down] = self.elements.z[down] - dz # move to layer underneath

        # put the particles that belong to the surface slick (if present) back to the surface
        self.elements.z[surface] = 0.

        # Prevent elements to go below seafloor
        bottom = np.where(self.elements.z < Zmin)
        if len(bottom[0]) > 0:
            logging.debug('%s elements penetrated seafloor, lifting up' % len(bottom[0]))
            self.elements.z[bottom] = np.round(Zmin[bottom]/dz)*dz + dz/2.

        # Call surface interaction:
        # reflection at surface or formation of slick and wave mixing if implemented for this class
        self.surface_interaction(dt_mix)

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
    Condition_MLD=np.where(depth_t > self.environment.surface_boundary_layer)

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
            sigma_v=p.zeros(self.num_elements_active())
            sigma_u[Condition_MLD]=np.random.normal(0, 1,sum(Condition_MLD))*np.sqrt(Kh/self.time_step.total_seconds())
            sigma_v[Condition_MLD]=np.random.normal(0, 1,sum(Condition_MLD))*np.sqrt(Kh/self.time_step.total_seconds())
        else:
            sigma_u = np.random.normal(0, 1,self.num_elements_active())*np.sqrt(Kh/self.time_step.total_seconds()) ### According to Okubo, 1971
            sigma_v = np.random.normal(0, 1,self.num_elements_active())*np.sqrt(Kh/self.time_step.total_seconds()) ### According to Guizien et al. 2006

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

        Kh=0.0103*(gls**1.15)*(10**-4) # Equation from Okubo is giving Kh in cm/s2 --> conversion in m2/s by *

        if self.get_config('turbulentmixing:moduleturb')==3:
            sigma_u=np.zeros(self.num_elements_active())
            sigma_v=np.zeros(self.num_elements_active())
            sigma_u[Condition_MLD]=np.random.normal(0, 1,sum(Condition_MLD))*np.sqrt(Kh/self.time_step.total_seconds())
            sigma_v[Condition_MLD]=np.random.normal(0, 1,sum(Condition_MLD))*np.sqrt(Kh/self.time_step.total_seconds())
        else:
            sigma_u = np.random.normal(0, 1,self.num_elements_active())*np.sqrt(Kh/self.time_step.total_seconds()) ### According to Okubo, 1971
            sigma_v = np.random.normal(0, 1,self.num_elements_active())*np.sqrt(Kh/self.time_step.total_seconds()) ### According to Guizien et al. 2006


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
            sigma_v=p.zeros(self.num_elements_active())
            sigma_u[Condition_MLD]=np.random.normal(0, np.sqrt(TKE*2/3),sum(Condition_MLD))*self.time_step.total_seconds()
            sigma_v[Condition_MLD]=np.random.normal(0, np.sqrt(TKE*2/3),sum(Condition_MLD))*self.time_step.total_seconds()
        else:
            sigma_u = np.random.normal(0, np.sqrt(TKE*2/3),self.num_elements_active())*self.time_step.total_seconds()
            sigma_v = np.random.normal(0, np.sqrt(TKE*2/3),self.num_elements_active())*self.time_step.total_seconds()

    self.update_positions(sigma_u, sigma_v)

