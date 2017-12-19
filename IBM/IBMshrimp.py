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

import os
import numpy as np
from datetime import datetime, timedelta

from opendrift.models.opendrift3D import OpenDrift3DSimulation, Lagrangian3DArray
import Vertical_module as vp ### Functions to use vertical and horizontal diffusivity
from opendrift.elements import LagrangianArray


# Defining the larvaes element properties
class PelagicShrimp(Lagrangian3DArray):
    """Extending Lagrangian3DArray with specific properties for pelagic eggs
    """

    variables = LagrangianArray.add_variables([
        ('diameter', {'dtype': np.float32,
                      'units': 'm',
                      'default': 0.0003}),  # size eggs shrimps
        ('density', {'dtype': np.float32,
                     'units': 'kg/m^3',
                     'default': 1028.}),  # density eggs/larvae of shrimp
        ('age_seconds', {'dtype': np.float32,
                         'units': 's',
                         'default': 0.}),
        ('duration', {'dtype': np.float32,  # Duration of the crustacean stages
                      'units': 's',
                      'default': 1.74*86400}),
        ('stages_end_buoyancy', {'dtype': np.int16, # Precise the stage when the buoyancy is no more efficient
                                 'units': '[]',
                                 'default': 2}),
        ('TotalLength',{'dtype':np.float32,  # Size of the Post Larvae to use for the swimming /!\ could be explored for younger stages
                        'units':'mm',
                        'default':4.26}),
        ('particles_velocity',{'dtype':np.float32, # Swimming velocity used or not according to the configuration (see later)
                        'units':'m/s',
                        'default':0.}),
        ('stages', {'dtype': np.int16, # Stages of the larvae (0 = eggs, e.g.:0=eggs, 1= Nauplius, 2=Protozoea, 3=Mysis, 4= Postlarvae)
                    'units': '[]',
                    'default': 0})])

# Function to change the stages of the larvae if the 'duration' is lower than the time of drift
    def updateEggStages(self):
        self.elements.stages[self.elements.age_seconds >= self.elements.duration] += 1
        self.elements.stages[self.elements.stages>=4] = 4

# Function to compute the duration of the stages according to Temperature of Water. The new duration will be added to the previous,
# in order to have UpdateEggStages working well.
    def updateEggDuration(self):
        Temperature=self.environment.sea_water_temperature[np.all([self.elements.age_seconds >= self.elements.duration,self.elements.stages < 4],0)]
        B=np.array([1.59, 2.76, 3.77, 3.76])
        stages = np.int16(self.elements.stages[np.all([self.elements.age_seconds >= self.elements.duration,self.elements.stages < 4],0)])

        self.elements.duration[np.all([self.elements.age_seconds >= self.elements.duration,self.elements.stages < 4],0)] += (np.exp(-0.076 * Temperature +B[stages]))*86400 # Equation can be modified to fit other coefficients in case of PLD = a+b*Temperature. Could be modified to include Salinity



class PelagicShrimpDrift(OpenDrift3DSimulation, PelagicShrimp):
    """Buoyant particle trajectory model based on the OpenDrift framework.

        Developed at ICM-CSIC (SPAIN), from the original script of PelagicEgg developped at MET, Norway

        Generic module for particles that are subject to vertical or horizontal turbulent
        mixing with the possibility for positive or negative buoyancy
        Based on larval development of decapods (Penaeids)

        Under construction.
    """

    ElementType = PelagicShrimp

    required_variables = ['x_sea_water_velocity', 'y_sea_water_velocity',
                          'sea_surface_height',
                          'land_binary_mask',
                          'sea_floor_depth_below_sea_level',
                          'ocean_vertical_diffusivity',
                          'sea_water_temperature',
                          'sea_water_salinity',
                          'surface_downward_x_stress',
                          'surface_downward_y_stress',
                          'turbulent_kinetic_energy',
                          'turbulent_generic_length_scale',
                          'upward_sea_water_velocity',
                          'ocean_horizontal_diffusivity', ### Added
                          'surface_boundary_layer'
                          ]

    # Vertical profiles of the following parameters will be available in
    # dictionary self.environment.vertical_profiles
    # E.g. self.environment_profiles['x_sea_water_velocity']
    # will be an array of size [vertical_levels, num_elements]
    # The vertical levels are available as
    # self.environment_profiles['z'] or
    # self.environment_profiles['sigma'] (not yet implemented)
    
    required_profiles = ['sea_water_temperature','sea_water_salinity','ocean_vertical_diffusivity']#,'ocean_horizontal_diffusivity']
    required_profiles_z_range = [-2000, 0]  # The depth range (in m) which
                                          # profiles shall cover

    fallback_values = {'x_sea_water_velocity': 0,
                       'y_sea_water_velocity': 0,
                       'land_binary_mask' : 0,
                       'sea_floor_depth_below_sea_level': 10,
                       'ocean_vertical_diffusivity': 0.002,  # m2.s-1
                       'sea_water_temperature': 12.,
                       'sea_water_salinity': 37.,
                       'surface_downward_x_stress': 0,
                       'surface_downward_y_stress': 0,
                       'turbulent_kinetic_energy': 0,
                       'turbulent_generic_length_scale': 0,
                       'upward_sea_water_velocity': 0,
                       'ocean_horizontal_diffusivity':1., # m2.s-1
                       'surface_boundary_layer': -50. # m
                       }

    # Configuration
    configspec = '''
        [drift]
            scheme = string(default='runge kunta')
        [processes]
            turbulentmixing = boolean(default=False)
            verticaladvection = boolean(default=True)
            verticalmodule = boolean(default=False)
            buoyancyparticles = boolean(default=True)
            Vlimparticles = boolean(default=False)
        [turbulentmixing]
            timestep = float(min=0.1, max=3600, default=1.)
            verticalresolution = float(min=0.01, max=10, default = 1.)
            diffusivitymodel = string(default='environment')
            moduleturb'= integer(min=0,max=3,default=0)
            diffusivityhorizontal= option('constant','environnemnt',default='constant')
    '''


    def __init__(self, *args, **kwargs):

        # Calling general constructor of parent class
        super(PelagicShrimpDrift, self).__init__(*args, **kwargs)
        self._add_config('processes:verticalmodule', 'boolean(default=False)', comment='Buoyancy and swimming')
        self._add_config('processes:Settlement', 'boolean(default=False)', comment='Settlement swimming')
        self._add_config('processes:Buoyancy', 'boolean(default=False)', comment='Buoyancy only')
        self._add_config('processes:horiz_turbulent','boolean(default=False)',comment='Horizontal diffusivity')
        self._add_config('turbulentmixing:moduleturb','integer(min=0,max=4,default=0)',comment='Module Diffusivity 0,1,2,3,4')
        self._add_config('turbulentmixing:diffusivityhorizontal','option(constant,environnment,default=constant)',comment='Module Diffusivity')

    def VerticalSettlementSwimming(self):
    # Computing the velocity of PL to go down according to the size of the PL: velocity is Total Length/s
        Larvaestages= self.elements.stages
        #
        for ind in xrange(len(self.elements.lat)):
            if Larvaestages[ind]==4:
                self.elements.particles_velocity[ind] = -self.elements.TotalLength[ind]*(10**-3) # m/s
    


    def update_terminal_velocity(self, Tprofiles=None, Sprofiles=None, z_index=None):
        """Calculate terminal velocity for Pelagic Egg
        according to
        S. Sundby (1983): A one-dimensional model for the vertical distribution
        of pelagic fish eggs in the mixed layer
        Deep Sea Research (30) pp. 645-661
        Method copied from ibm.f90 module of LADIM:
        Vikebo, F., S. Sundby, B. Aadlandsvik and O. Otteraa (2007),
        Fish. Oceanogr. (16) pp. 216-228
        """
        g = 9.81  # ms-2
        
        # Pelagic Egg properties that determine buoyancy
        eggsize = self.elements.diameter  # 0.0003 for Shrimp
        eggdensity= self.elements.density
                   
        T0 = self.environment.sea_water_temperature
        S0 = self.environment.sea_water_salinity

        # The density difference bettwen a pelagic egg and the ambient water
        # is regulated by their salinity difference through the
        # equation of state for sea water.
        # The Egg has the same temperature as the ambient water and its
        # salinity is regulated by osmosis through the egg shell.
        DENSw = self.sea_water_density(T=T0, S=S0)
        DENSegg = eggdensity
        dr = DENSw-DENSegg  # density difference

        # water viscosity
        my_w = 0.001*(1.7915 - 0.0538*T0 + 0.007*(T0**(2.0)) - 0.0023*S0)
        # ~0.0014 kg m-1 s-1
            
        # terminal velocity for low Reynolds numbers
        W = (1.0/my_w)*(1.0/18.0)*g*eggsize**2 * dr
    
        #check if we are in a Reynolds regime where Re > 0.5
        highRe = np.where(W*1000*eggsize/my_w > 0.5)

        # Use empirical equations for terminal velocity in
        # high Reynolds numbers.
        # Empirical equations have length units in cm!
        my_w = 0.01854 * np.exp(-0.02783 * T0)  # in cm2/s
        d0 = (eggsize * 100) - 0.4 * \
        (9.0 * my_w**2 / (100 * g) * DENSw / dr)**(1.0 / 3.0)  # cm
        W2 = 19.0*d0*(0.001*dr)**(2.0/3.0)*(my_w*0.001*DENSw)**(-1.0/3.0)
        # cm/s
        W2 = W2/100.  # back to m/s
        W[highRe] = W2[highRe]
    
        W[self.elements.stages >= self.elements.stages_end_buoyancy]=0
        self.elements.terminal_velocity = W

    def update(self):
        """Update positions and properties of particles."""

        # Update element age
        self.elements.age_seconds += self.time_step.total_seconds()

        # Turbulent Mixing
        self.update_terminal_velocity() ### Buoyancy velocity updated
        self.VerticalSettlementSwimming() ### Settlement velocity updated, if module activated
        vp.vertical_mixing(self) # will be accounted if turbulent mixing is TRUE
        vp.vertical_module(self) # will be accounted if verticalmodule is TRUE
        vp.horizontal_mixing(self,'output')
        
        # Plankton development
        self.updateEggStages()
        self.updateEggDuration()

        # Horizontal advection
        self.advect_ocean_current()
       
        # Vertical advection
        if self.get_config('processes:verticaladvection') is True:
            self.vertical_advection()

