import random
from random_timestamp import random_timestamp
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

random.seed(30)
task_distribution = []
for i in range(1000):
    date_time = random_timestamp(2020, 4, 16, 'DATE/TIME')
    cs_name = random.randint(1,5)
    cpu_cycles = random.randint(1,10)
    energy_units = random.randint(1,10)
    network_bandwidth = random.randint(1,10)
    storage_space = random.randint(1,10)
    duration = round((cpu_cycles + energy_units + network_bandwidth + storage_space)+
    (cpu_cycles + energy_units + network_bandwidth + storage_space)*random.uniform(0.3,1),0)
    price = round(random.uniform((cpu_cycles + energy_units + network_bandwidth + storage_space)*5,(cpu_cycles + energy_units + network_bandwidth + storage_space)*10), 2)
    task_distribution.append([date_time,cs_name,cpu_cycles,energy_units,network_bandwidth,storage_space,duration, price])
    
df = pd.DataFrame(task_distribution, columns=['date_time', 'cs_name', 'cpu_cycles',
'energy_units', 'network_bandwidth', 'storage_space', 'duration', 'price'])