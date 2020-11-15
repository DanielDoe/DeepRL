import random
from random_timestamp import random_timestamp

def task_generator():
    task_distribution = []
    for i in range(10):
        date_time = random_timestamp(2020, 4, 16, 'DATE/TIME')
        node_name = random.randint(1,5)
        cpu_cycles = random.randint(1,10)
        energy_units = random.randint(1,10)
        network_bandwidth = random.randint(1,10)
        storage_space = random.randint(1,10)
        total_req = cpu_cycles + energy_units + network_bandwidth + storage_space
        price = round(random.uniform(300,1000), 1)
        task_distribution.append([date_time,node_name,cpu_cycles,energy_units,network_bandwidth,storage_space,total_req,price])
        
    return task_distribution
        