import os
import sys

import torch.cuda

from agent import *
from env import *
from model import reset_model_count
import random
from utils import get_logger
from utils.data_process import four_season_train_test_split, get_scaled_data, get_scaled_data_wind_power


def train(data, season, search_type, place):
    from utils import NASConfig
    EPISODE = NASConfig['EPISODE']
    IterationEachTime = NASConfig['IterationEachTime']
    NASConfig['OUT_DIR'] = f"../bin_rnn/{place}_{search_type}_episode-{EPISODE}-{IterationEachTime}-{season}"
    NASConfig['SQL_FILE'] = f"{NASConfig['OUT_DIR']}/model.db"
    # NASConfig['timeLength'] = 168
    # NASConfig['EPISODE'] = 200
    if not os.path.exists(NASConfig['OUT_DIR']):
        os.makedirs(NASConfig['OUT_DIR'])
    reset_model_count()
    logger = get_logger(NASConfig['OUT_DIR'])
    env = NasEnv(NASConfig['NetPoolSize'], data)
    agent = AgentREINFORCE(16, 50, NASConfig['MaxLayers'])
    net_pool = env.reset()
    for i in range(NASConfig['EPISODE']):
        try:
            action = agent.get_action(net_pool)
            net_pool, reward, done, info = env.step(action)
            agent.update(reward, action, net_pool)
            env.render()
            torch.cuda.empty_cache()
            logger.fatal(f'episode {i} finish,\tpool {len(env.net_pool)},\tperformance:{env.performance()}\t')
        except Exception as e:
            logger.fatal(f'{e}\n')
            logger.fatal(str(e))
            print(e)


def train_eas(data, season, search_type, place, wide_time = 3, deep_time = 3):
    from utils import NASConfig
    EPISODE = NASConfig['EPISODE']
    IterationEachTime = NASConfig['IterationEachTime']
    NASConfig['OUT_DIR'] = f"../bin_eas/{place}_{search_type}_episode-{EPISODE}-{IterationEachTime}-{season}"
    NASConfig['SQL_FILE'] = f"{NASConfig['OUT_DIR']}/model.db"
    # NASConfig['timeLength'] = 168
    # NASConfig['EPISODE'] = 100
    if not os.path.exists(NASConfig['OUT_DIR']):
        os.makedirs(NASConfig['OUT_DIR'])
    reset_model_count()
    logger = get_logger(NASConfig['OUT_DIR'])
    env = NasEnv(NASConfig['NetPoolSize'], data)
    agent = AgentREINFORCE(16, 50, NASConfig['MaxLayers'])
    net_pool = env.reset()
    for i in range(NASConfig['EPISODE']):
        if len(net_pool) == 0:
            break
        actions = []
        for d in range(deep_time):
            action = agent.get_action(env.net_pool)
            env.step_eas(action,'deeper')
            actions.append(action)
        for w in range(wide_time):
            action = agent.get_action(env.net_pool)
            env.step_eas(action, 'wider')
            actions.append(action)
        env.net_pool.extend(net_pool)
        env.net_pool = list(set(env.net_pool))
        net_pool, reward, done, info = env.step_eas(None, 'render')
        for action in actions:
            agent.update(reward, action, net_pool)
        env.render()
        torch.cuda.empty_cache()
        logger.fatal(f'episode {i} finish, pool {len(env.net_pool)}, performance:{env.performance()}')


def train_wind_power(data, season, search_type, place):
    from utils import NASConfig
    EPISODE = NASConfig['EPISODE']
    IterationEachTime = NASConfig['IterationEachTime']
    NASConfig['OUT_DIR'] = f"../bin_wind/{place}_{search_type}_episode-{EPISODE}-{IterationEachTime}-{season}"
    NASConfig['SQL_FILE'] = f"{NASConfig['OUT_DIR']}/model.db"
    NASConfig['timeLength'] = 72
    NASConfig['EPISODE'] = 100
    if not os.path.exists(NASConfig['OUT_DIR']):
        os.makedirs(NASConfig['OUT_DIR'])
    reset_model_count()
    logger = get_logger(NASConfig['OUT_DIR'])
    env = NasEnv(NASConfig['NetPoolSize'], data)
    agent = AgentREINFORCE(16, 50, NASConfig['MaxLayers'])
    net_pool = env.reset()
    for i in range(NASConfig['EPISODE']):
        action = agent.get_action(net_pool)
        net_pool, reward, done, info = env.step(action)
        agent.update(reward, action, net_pool)
        env.render()
        torch.cuda.empty_cache()
        logger.fatal(f'episode {i} finish,\tpool {len(env.net_pool)},\tperformance:{env.performance()}\ttop performance:{env.top_performance()}')


def train_wind_power_eas(data, season, search_type, place, wide_time = 3, deep_time = 3):
    from utils import NASConfig
    EPISODE = NASConfig['EPISODE']
    IterationEachTime = NASConfig['IterationEachTime']
    NASConfig['OUT_DIR'] = f"../bin_wind_eas/{place}_{search_type}_episode-{EPISODE}-{IterationEachTime}-{season}"
    NASConfig['SQL_FILE'] = f"{NASConfig['OUT_DIR']}/model.db"
    NASConfig['SQL_FILE'] = f"{NASConfig['OUT_DIR']}/model.db"
    NASConfig['timeLength'] = 72
    NASConfig['EPISODE'] = 100
    if not os.path.exists(NASConfig['OUT_DIR']):
        os.makedirs(NASConfig['OUT_DIR'])
    reset_model_count()
    logger = get_logger(NASConfig['OUT_DIR'])
    env = NasEnv(NASConfig['NetPoolSize'], data)
    agent = AgentREINFORCE(16, 50, NASConfig['MaxLayers'])
    net_pool = env.reset()
    for i in range(NASConfig['EPISODE']):
        if len(net_pool) == 0:
            break
        actions = []
        for d in range(deep_time):
            action = agent.get_action(env.net_pool)
            env.step_eas(action,'deeper')
            actions.append(action)
        for w in range(wide_time):
            action = agent.get_action(env.net_pool)
            env.step_eas(action, 'wider')
            actions.append(action)
        env.net_pool.extend(net_pool)
        env.net_pool = list(set(env.net_pool))
        net_pool, reward, done, info = env.step_eas(None, 'render')
        for action in actions:
            agent.update(reward, action, net_pool)
        env.render()
        torch.cuda.empty_cache()
        logger.fatal(f'episode {i} finish,\tpool {len(env.net_pool)},\tperformance:{env.performance()}\ttop performance:{env.top_performance()}')


if __name__ == '__main__':
    random_seed = 33243242
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data_root = '../data'
    place_codes = ['WF1', 'WF2']
    for place in place_codes:
        spring, summer, autumn, winter = get_scaled_data_wind_power(data_root, place, NASConfig['timeLength'])
        train_type = 'rnn'
        train_func = train_wind_power
        train_func(spring, 'spring', train_type, place)
        train_func(summer, 'summer', train_type, place)
        train_func(autumn, 'autumn', train_type, place)
        train_func(winter, 'winter', train_type, place)
    place_codes = ['NH', 'ME']
    for place in place_codes:
        X, y = get_scaled_data(data_root, place, NASConfig['timeLength'])
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        y = y.reshape((-1, 1))
        X = X.permute(0, 2, 1)
        # X = X[:,0:1,:]
        spring, summer, autumn, winter = four_season_train_test_split(X, y)

        train_type = 'rnn'
        train_func = train
        train_func(spring, 'spring', train_type, place)
        train_func(summer, 'summer', train_type, place)
        train_func(autumn, 'autumn', train_type, place)
        train_func(winter, 'winter', train_type, place)


