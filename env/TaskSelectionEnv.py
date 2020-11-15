import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

MAX_CPU_CYCLES = 8
MAX_ENERGY_UNITS = 8
MAX_BANDWIDTH = 8
MAX_STORAGE = 8
MAX_ALLOWED_TASK = 2
MAX_STEPS = 100
MAX_PRICE = 1000

INITIAL_REWARD_BALANCE = 0



class TaskSelectionEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TaskSelectionEnv, self).__init__()

        self.df = df
        #self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format accept x%, reject x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # tasks contains the requirements for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7, 7), dtype=np.float16)
        
        print(self.action_space)
        print(self.observation_space)
    
    def _next_task_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'cpu_cycles'].values / MAX_CPU_CYCLES,
            self.df.loc[self.current_step: self.current_step +
                        5, 'energy_units'].values / MAX_ENERGY_UNITS,
            self.df.loc[self.current_step: self.current_step +
                        5, 'network_bandwidth'].values / MAX_BANDWIDTH,
            self.df.loc[self.current_step: self.current_step +
                        5, 'storage_space'].values / MAX_STORAGE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'price'].values / MAX_PRICE,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current task to a random task within the dataset step
        current_task = random.randint(
            self.df.loc[self.current_step, "cpu_cycles"],
            self.df.loc[self.current_step, "energy_units"],
            self.df.loc[self.current_step, "network_bandwidth"], 
            self.df.loc[self.current_step, "storage_space"])

        action_type = action[0]
        amount = action[1]
        
        #Check the utilty function to know how good or bad a task is.

        if action_type < 1:
            print("Accept task")

        elif action_type < 2:
           print("Reject task")
            
            
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.cpu_cycles = MAX_CPU_CYCLES
        self.energy_units = MAX_ENERGY_UNITS
        self.network_bandwidth = MAX_BANDWIDTH
        self.storage_space = MAX_STORAGE

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'date_time'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        
        
