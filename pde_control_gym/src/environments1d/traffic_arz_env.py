import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
from pde_control_gym.src.environments1d.traffic_arz_utils import Veq, F_r, F_y

class TrafficPDE1D(PDEEnv1D):
    r""" 
    Traffic ARZ PDE

    This class implements the Traffic ARZ PDE and inhertis from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv1D` in conjunction with the arguments presented here

    :param simulation_type: Defines the type of boundary control. Inputs 'inlet', 'outlet' and 'both' represents boundary control at inlet, outlet and both respectively. 
    :param v_max: Maximum permissible velocity (meters/second) on freeway under simulation 
    :param ro_max: Maximum permissible density (vehicles/meter) on freeway under simulation
    :param v_steady: Desired steady state velocity (meters/second). Ensure that v_steady and ro_steady obey the equilibrium equation v_steady = v_max(1 - ro_steady/v_max)
    :param ro_steady: Desired steady state density (vehicles/meter). Ensure that v_steady and ro_steady obey the equilibrium equation v_steady = v_max(1 - ro_steady/v_max)
    :param tau: Relaxation time (seconds) required by the driver to adjust to the new velocity
    """
    def __init__(self, 
                 simulation_type: str = 'inlet', 
                 v_steady: float = 10,
                 ro_steady: float = 0.12,
                 v_max: float = 40,
                 ro_max: float = 0.16,
                 tau: float = 60,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.simulation_type = simulation_type
        self.vm = v_max
        self.rm = ro_max
        self.qm = v_max * ro_max/4
        self.tau = tau

        if v_steady != Veq(v_max, ro_max, ro_steady):
            raise ValueError('The steady state velocity and density do not satisfy the equilibrium condition. Check the values of v_steady and ro_steady and ensure that they obey v_steady = v_max(1 - ro_steady/v_max).')
        self.vs = v_steady
        self.rs = ro_steady

        self.qs = v_steady * ro_steady
        self.ps = self.vm/self.rm * self.qs/self.vs
        
        if self.simulation_type == 'outlet':
            print('Case 1: Outlet Boundary Control')
        elif self.simulation_type == 'inlet':
            print('Case 2: Inlet Boundary Control')
        elif self.simulation_type == 'both':
            print('Case 3: Outlet & Inlet Boundary Control')
        else:
            raise ValueError('Invalid simulation type')      
    
        x = np.arange(0,self.X+self.dx,self.dx)
        self.L = self.X
        self.M = len(x)
        self.qs = self.qs
        self.qs_input = np.linspace(self.qs/2,2*self.qs,40)
        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])


        #Initial condition of the PDE
        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)
        
        self.info = dict()
        self.info['V'] = self.v

	    # Observation space
        self.observation_space = spaces.Box(low=0, high=40, shape=(2 * self.M,), dtype="float64")

        #Action space
        if self.simulation_type == 'both':
            self.action_space = spaces.Box(dtype=np.float64, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(2,))
        else:
            self.action_space = spaces.Box(dtype=np.float64, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(1,))
            



    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if (self.time_index >= self.T / self.dt):
            self.time_index = 0
            return True
        else:
            return False

    def truncate(self):
        """
        truncate 

        Determines whether to truncate the episode based on the PDE state size and the vairable ``limit_pde_state_size`` given in the PDE environment intialization.
        """
        if all(self.r - self.rs == 0) and all(self.v - self.vs == 0):
            return True
        else:
            return False


    def step(self, action):
        """
        step

        Updates the PDE state based on the action taken and returns the new state, reward, done, truncated and info. The PDE is solved using finite differencing explained in docs and the reward is computed based on the deviation from the desired density and velocity.

        :param action: The control input to apply to the freeway at the inlet, outlet or both.
        :return: A tuple of:
            - observation (np.ndarray): The concatenated state vector containing density (`r`) and velocity (`v`) after taking action.
            - reward (float): The reward computed based on deviation from desired density and velocity after action.
            - done (bool): Whether the simulation should terminate.
            - truncated (bool): Whether the simulation was truncated as desired state has been achieved.
            - info (dict): Additional information about the current state for debugging.
        """
        Nx = self.nx
        dx = self.dx
        dt = self.dt
        self.time_index += dt
        qs_input = action
        
        if self.simulation_type == 'both':
            qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)
            q_inlet_input = qs_input[0]
            q_outlet_input = qs_input[1]

        else:
            qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)[0]

        #PDE control at inlet
        if self.simulation_type == 'outlet':
			# Fixed inlet boundary input
            self.q_inlet = self.qs

        elif self.simulation_type == 'inlet':
			# Control inlet boundary 
            self.q_inlet = qs_input

        elif self.simulation_type == 'both':
            # Control inlet boundary 
            self.q_inlet = q_inlet_input

        # Boundary conditions
        self.r[0] = self.r[1]
        self.y[0] = self.q_inlet - self.r[0] * Veq(self.vm, self.rm, self.r[0])
        self.r[self.M-1] = self.r[self.M-2]

        # PDE control at outlet
        if self.simulation_type == 'outlet':
            # Control outlet boundary 
            self.y[self.M-1] = qs_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        elif self.simulation_type == 'inlet':
            # Fixed outlet boundary 
            self.y[self.M-1] = self.qs - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        elif self.simulation_type == 'both':
            # Control outlet boundary 
            self.y[self.M-1] = q_outlet_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        #Finite differencing of PDEs
        for j in range(1,self.M-1) :

            r_pmid = 1/2 * (self.r[j+1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_r(self.vm, self.rm, self.r[j], self.y[j]) )

            y_pmid = 1/2 * (self.y[j+1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_y(self.vm, self.rm, self.r[j], self.y[j])) - 1/4 * dt / self.tau * (self.y[j+1]+self.y[j])

            r_mmid = 1/2 * (self.r[j-1 ] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j], self.y[j]) - F_r(self.vm, self.rm, self.r[j-1], self.y[j-1]))

            y_mmid = 1/2 * (self.y[j-1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j], self.y[j]) - F_y(self.vm, self.rm, self.r[j-1], self.y[j-1])) - 1/4 * dt / self.tau * (self.y[j-1]+self.y[j])

            self.r[j] = self.r[j] - dt/dx * (F_r(self.vm, self.rm, r_pmid, y_pmid) - F_r(self.vm, self.rm, r_mmid, y_mmid))
            self.y[j] = self.y[j] - dt/dx * (F_y(self.vm, self.rm, r_pmid, y_pmid) - F_y(self.vm, self.rm, r_mmid, y_mmid)) - 1/2 * dt/self.tau * (y_pmid + y_mmid)

        # Calculate Velocity
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)

        reward = self.reward_class.reward(self.vs, self.rs, self.v, self.r)
        
        return np.reshape(np.concatenate((self.r, self.v)), -1), reward, self.terminate(), self.truncate(), self.info


    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        Resets the environment to an initial state and returns an initial observation and info.
    
        :param seed: Optional seed for reproducibility.
        :param options: Optional dictionary to define initialization options.
        :return: A tuple of (observation, info).
        """

        x = np.arange(0,self.X+self.dx,self.dx)
        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])


        #Initial condition of the PDE
        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)

        obs = np.reshape(np.concatenate((self.r, self.v)), -1)
    
        info = {}  # Optional info dict for debugging/logging
    
        return obs, info



