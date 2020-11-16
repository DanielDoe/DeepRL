import warnings
import numpy as np
import pandas as pd
import time
import sys
# local library
import task_utils
import ddpg_dqn_pricing_config

def main(arg):
    input_data = task_utils.df

    train_input = input_data[:700]
    test_input = input_data[700:]
    
    # training
    n_task = len(train_input)
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
            print('portfolio:', action)
            print('profit:', prof)
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
    