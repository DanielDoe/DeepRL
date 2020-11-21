import warnings
import numpy as np
import pandas as pd
import time
import sys
# local library
from task_utils import task_generator
import ddpg_dqn_pricing_config

#print(df[:700].values[0])

def main(arg):
    df = pd.DataFrame(task_generator(), columns=['date_time', 'cs_name', 'cpu_cycles',
'energy_units', 'network_bandwidth', 'storage_space', 'duration', 'price'])
    df.drop(['date_time', 'cs_name'], axis=1, inplace=True)
    train_input = df[:700]
    test_input = df[700:]
    
    # training
    n_task = len(train_input.values[0])
    sys.path.append("./model")
    print(arg)


    if arg == "ddpg":
        from ddpg_model import DDPG
        from ddpg_dqn_pricing_config import DDPGConfig
        config = DDPGConfig(n_task)
        ddpg = DDPG(config)
        values = ddpg.train(train_input)
    elif arg == "dqn":
        from dqn_model import DQN
        from ddpg_dqn_pricing_config import DQNConfig
        config = DQNConfig(n_task)
        dqn = DQN(config)
        values = dqn.train(train_input)
        return values
    else:
        return None
    
    # prediction
    price = []
    date = []
    index = test_input.index
    values = test_input.values
    old_value = values[0]
    prof = 0
    count = 0
    for i in range(1, len(index)):
        value = values[i]
        action = ddpg.predict_action(old_value)
        ddpg.update_memory(old_value, value)
        gain = np.sum((value - old_value) * action)
        prof += gain
        price.append(prof)
        date.append(index[i])
        if count%10 == 0:
            result = pd.DataFrame(price, index=pd.DatetimeIndex(date))
            result.to_csv("test_result.csv")
        count += 1
        if count%10 == 0:
            print('time:', index[i])
            print('actions:', action)
            print('pred:', prof)
        print('***************************')
        for i in range(100):
            ddpg.update_weight()
        old_value = value
    result = pd.DataFrame(price, index=pd.DatetimeIndex(date))
    return result
    
if __name__ == '__main__':
    arg = sys.argv[1]
    warnings.filterwarnings("ignore")
    result = main(arg)
    