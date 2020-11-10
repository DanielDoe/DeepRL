import gym
import json
import datetime as dt
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env.StockTradingEnv import REWARD_VALUES
from data.task_distribution import task_generator

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')
print(df)
df2 = pd.DataFrame(task_generator(), columns=['date_time', 'cpu_cycles',
'energy_units', 'network_bandwidth', 'storage_space', 'total_req', 'price'])
print(df2.sort_values('date_time'))

'''
# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=500)

obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

df = pd.DataFrame(REWARD_VALUES) 
df.plot.line() 
plt.show()
#print(f'Rewards: {REWARD_VALUES}')
'''
