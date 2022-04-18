import random

import gym
import numpy as np
import torch.nn

from model import generate_new_model_config
from utils import get_logger, NASConfig
import torchinfo

logger = get_logger('NAS_ENV')


class NasEnv(gym.Env):
    """
    NAS environment , the NAS search action and evaluation is in this class.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self,pool_size, train_test_data):
        """
        @param pool_size:
        @param train_test_data:  (X_train, y_train, X_test, y_test), for input X, the dimension should be
        (batch, feature, time)
        """
        self.train_test_data = train_test_data
        self.net_pool = None
        self.pool_size = pool_size
        self.best_performance = 1e9  # record best performance
        self.global_train_times = 0
        self.no_enhancement_episodes = 0  # record episodes that no enhancement of  prediction perfromance
        if NASConfig['GPU']:
            self.train_test_data = []
            for i in range(len(train_test_data)):
                self.train_test_data.append(train_test_data[i].cuda())


    def step(self, action):
        net_pool = []
        for i in range(len(self.net_pool)):
            net = self.net_pool[i]
            action_i = action['action'][i]
            select = action_i['select']
            if net.train_times < 1000:
                net_pool.append(net)
                logger.info(f"net index {i}-{net.index} : continue training({net.train_times})")
            # select representations
            if select == 0:  # do nothing
                logger.info(f"net index {i}-{net.index} :do not change network")
                continue
            if select == 1:  # wider the net
                logger.info(f"net index {i}-{net.index} :wider the net")
                net = net.perform_wider_transformation(action_i['wider'])
                net_pool.append(net)
            if select == 2:  # deeper the net
                logger.info(f"net index {i}-{net.index} :deeper the net")
                net = net.perform_deeper_transformation(action_i['deeper'])
                if len(net.model_config.modules) <= NASConfig['MaxLayers']:  # constrain the network's depth
                    net_pool.append(net)
        for net in net_pool:
            net.add_noise(0.01)
        self.net_pool = net_pool
        X_train, y_train, X_test, y_test = self.train_test_data
        feature_shape = X_train.shape[1:]
        base_structure = [  # manual design
            ['rnn'],
            ['conv'],
            ['dense'],
            ['conv','rnn','dense'],
            ['rnn','rnn','rnn','dense'],
            ['conv','conv','dense','dense'],
        ]
        skeleton = random.sample(base_structure, 1)[0]
        self.net_pool.append(generate_new_model_config(feature_shape, 1, skeleton).generate_model())
        self.net_pool = list(set(self.net_pool))
        self._train_and_test()
        self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
        self.net_pool = self.net_pool[:self.pool_size]
        observation = self.net_pool
        reward = self.get_reward()
        done = True
        info = {}
        return observation, reward, done, info

    def step_eas(self, action, action_type):
        if action_type == 'render':
            self._train_and_test()
            self.net_pool = sorted(self.net_pool, key=lambda x: x.test_loss)
            self.net_pool = self.net_pool[:self.pool_size]
            observation = self.net_pool
            reward = self.get_reward()
            done = True
            info = {}
            return observation, reward, done, info

        net_pool = []
        for i in range(len(self.net_pool)):
            net = self.net_pool[i]
            action_i = action['action'][i]
            if action_type == 'wider' and net.model_config.can_widen:
                logger.info(f"net index {i}-{net.index} :wider the net")
                net = net.perform_wider_transformation(action_i['wider'])
                net_pool.append(net)
            elif action_type == 'wider' and not net.model_config.can_widen:
                net_pool.append(net)  # do not lose the max level net
            elif action_type == 'deeper':
                logger.info(f"net index {i}-{net.index} :deeper the net")
                net = net.perform_deeper_transformation(action_i['deeper'])
                if len(net.model_config.modules) <= NASConfig['MaxLayers']:
                    net_pool.append(net)
        for net in net_pool:
            net.add_noise(0.01)
        self.net_pool = net_pool
        self.net_pool = list(set(self.net_pool))

    def random_setp(self):
        X_train, y_train, X_test, y_test = self.train_test_data
        feature_shape = X_train.shape[1:]
        self.net_pool.append(generate_new_model_config(feature_shape, 1).generate_model())
        self.net_pool = list(set(self.net_pool))
        self._train_and_test()
        self.net_pool = sorted(self.net_pool,key = lambda x: x.test_loss_best)
        self.net_pool = self.net_pool[:self.pool_size]
        observation = self.net_pool
        reward = self.get_reward()
        done = True
        info = {}
        return observation, reward, done, info

    def performance(self):
        return np.mean([net.test_loss_best for net in self.net_pool])

    def top_performance(self):
        return np.min([net.test_loss_best for net in self.net_pool])

    def get_reward(self):
        reward = []
        for net in self.net_pool:
            reward.append(1 / net.test_loss)
        return reward

    def reset(self):
        """
        reset all net to random init, this should call once in whole program
        train one round before return netpool
        :return:  net pool containing all network under searching
        """
        X_train, y_train, X_test, y_test = self.train_test_data
        feature_shape = X_train.shape[1:]
        self.net_pool = [
            generate_new_model_config(feature_shape, 1, ['rnn']).generate_model(),
            generate_new_model_config(feature_shape, 1, ['conv']).generate_model(),
            generate_new_model_config(feature_shape, 1, ['dense']).generate_model(),
                         ]
        # self.net_pool = [generate_new_model_config(feature_shape, 1).generate_model()
        #                  for i in range(self.pool_size)]
        self._train_and_test()
        self.render()
        return self.net_pool

    def render(self, mode='human'):
        X_train, y_train, X_test, y_test = self.train_test_data
        for net in self.net_pool:
            if NASConfig['GPU']:
                net.to_cuda()
            net.save_model()
            net.save_pred_result(X_test, y_test)
            if NASConfig['GPU']:
                net.to_cpu()

    def _train_and_test(self):
        X_train, y_train, X_test, y_test = self.train_test_data
        self.global_train_times += NASConfig['IterationEachTime']
        for net in self.net_pool:
            if NASConfig['GPU']:
                net.to_cuda()
            net.train(X_train, y_train)
            net.test(X_test, y_test)

        self.render()  # save train result
