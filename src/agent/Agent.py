import numpy as np
from torch import nn
import torch
from torch.optim import Adam

from .components import *
from utils import get_logger

logger = get_logger('AgentREINFORCE')

class AgentREINFORCE:
    def __init__(self, input_size, hidden_size, max_layers):
        super().__init__()
        self.encoder_net = EncoderNet(input_size, hidden_size)
        self.wider_net = WinderActorNet(hidden_size * 2)
        self.deeper_net = DeeperActorNet(hidden_size * 2, max_layers)
        self.selector = SelectorActorNet(hidden_size * 2)
        self.selector_optimizer = Adam(self.selector.parameters())
        self.wider_optimizer = Adam(self.wider_net.parameters())
        self.deeper_optimizer = Adam(self.deeper_net.parameters())
        self.Categorical = torch.distributions.Categorical

    def to_cuda(self):
        self.wider_net.cuda()
        self.deeper_net.cuda()
        self.selector.cuda()
        self.encoder_net.cuda()

    def to_cpu(self):
        self.wider_net.cpu()
        self.deeper_net.cpu()
        self.selector.cpu()
        self.encoder_net.cpu()

    def get_action(self, net_pool):
        action, a_prob = [], []
        if NASConfig['GPU']:
            self.to_cuda()
        for net in net_pool:
            output, (h_n, c_n) = self.encoder_net.forward(net.model_config.token_list)
            select_action, select_prob = self.selector.get_action(h_n, net.model_config.can_widen)
            deeper_action, deeper_prob = self.deeper_net.get_action(h_n, net.model_config.insert_length)
            if net.model_config.can_widen:
                wider_action, wider_prob = self.wider_net.get_action(output, net.model_config.widenable_list)
            else:
                wider_action, wider_prob = None, None
            action.append({
                'select': select_action,
                'wider': wider_action,
                'deeper': deeper_action,
                'net_index': net.index
            })

            a_prob.append({
                'select': select_prob,
                'wider': wider_prob,
                'deeper': deeper_prob,
            })
        if NASConfig['GPU']:
            self.to_cpu()

        return {'action': action, 'prob': a_prob}

    def get_logprob_entropy(self, a_int, a_prob):
        if NASConfig['GPU']:
            a_int = a_int.cuda()
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int)

    def update(self, reward, action,net_pool):
        length = len(reward)
        if NASConfig['GPU']:
            self.to_cuda()
        for i in range(length):
            # calculate immediate net output
            net = net_pool[i]
            action_index = None
            for j in range(len(action['action'])):
                if action['action'][j]['net_index'] == net.index:
                    action_index = j
            if action_index is None or action_index >= length:
                continue
            logger.info(f'update net {net.index} action {action["action"][action_index]}')
            output, (h_n, c_n) = self.encoder_net.forward(net.model_config.token_list)
            select_action, select_prob = self.selector.get_action(h_n, net.model_config.can_widen)
            deeper_action, deeper_prob = self.deeper_net.get_action(h_n, net.model_config.insert_length)

            # update selector net
            action_onehot = torch.zeros(3)
            action_onehot[action['action'][action_index]['select']] = 1
            action_prob = select_prob
            self.update_one_subnet(self.selector_optimizer, action_onehot, action_prob, reward[action_index])

            # update wider net
            if action['prob'][action_index]['wider'] is not None and net.model_config.can_widen:
                wider_action, wider_prob = self.wider_net.get_action(output, net.model_config.widenable_list)
                action_onehot = torch.zeros(action['prob'][action_index]['wider'].shape[0])
                action_onehot[action['action'][action_index]['wider']] = 1
                action_prob = wider_prob
                self.update_one_subnet(self.wider_optimizer, action_onehot, action_prob, reward[action_index])

            # update deeper net type module
            action_onehot = torch.zeros(len(NASConfig['editable']))
            action_onehot[action['action'][action_index]['deeper'][0]] = 1
            action_prob = deeper_prob[0]
            prob = self.get_logprob_entropy(action_onehot, action_prob)
            type_loss = (- prob * reward[action_index]).sum()

            # update deeper net index module
            action_onehot = torch.zeros(NASConfig['MaxLayers'])
            action_onehot[action['action'][action_index]['deeper'][1]] = 1
            action_prob = deeper_prob[1]
            prob = self.get_logprob_entropy(action_onehot, action_prob)
            index_loss = (- prob * reward[action_index]).sum()

            # update deeper net by type and index loss
            deeper_loss = type_loss + index_loss
            self.deeper_optimizer.zero_grad()
            deeper_loss.backward()
            self.deeper_optimizer.step()
        if NASConfig['GPU']:
            self.to_cpu()

    def update_one_subnet(self, optimizer, onehot_action, action_prob, reward):
        prob = self.get_logprob_entropy(onehot_action, action_prob)
        policy_loss = - prob * reward
        policy_loss = policy_loss.mean()
        optimizer.zero_grad()
        policy_loss.backward(create_graph = True)
        optimizer.step()
