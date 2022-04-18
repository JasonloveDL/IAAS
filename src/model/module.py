import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from utils import modulesConfig, get_logger, NASConfig

logger = get_logger('module')


class NAS_Module:
    def __init__(self, name, input_shape):
        self.input_shape = input_shape  # (channel, feature)
        self.name = name
        self.output_shape = None
        self.params = None
        self.widenable = modulesConfig[name]['editable']
        self._module_instance = None  # buffer the instance
        self.current_level = None  # current width level
        self.widen_sample_fn = self.default_sample_strategy

    def __call__(self, x, *args, **kwargs):
        return self._module_instance(x)

    @property
    def is_max_level(self):
        if self.name == "dense":
            out_range = modulesConfig['dense']['out_range']
        elif self.name == 'conv':
            out_range = modulesConfig['conv']['out_channels']
        elif self.name == 'rnn':
            out_range = modulesConfig['rnn']['hidden_size']
        elif self.name == 'lstm':
            out_range = modulesConfig['lstm']['hidden_size']
        else:
            raise ValueError(f'module type of {self.name} have no level')
        return self.current_level >= max(out_range)

    @property
    def next_level(self):
        assert not self.is_max_level
        if self.name == "dense":
            out_range = modulesConfig['dense']['out_range']
        elif self.name == 'conv':
            out_range = modulesConfig['conv']['out_channels']
        elif self.name == 'rnn':
            out_range = modulesConfig['rnn']['hidden_size']
        elif self.name == 'lstm':
            out_range = modulesConfig['lstm']['hidden_size']
        else:
            raise ValueError(f'module type of {self.name} have no level')
        out_range = list(out_range)
        valid_range = []
        for i in out_range:
            if i > self.current_level:
                valid_range.append(i)
        return min(valid_range)

    def small_init_param(self, input_shape):
        """
        randomly generate parameters of this module
        :return: None
        """

        def get_level(level_list):
            # determine init level
            # return random.sample(level_list, 1)[0]
            return random.sample(level_list[:4], 1)[0]
        assert len(input_shape) == 2
        if self.name == "dense":
            out_range = modulesConfig['dense']['out_range']
            in_features = input_shape[1]
            self.current_level = get_level(out_range)
            self.params = {'in_features': in_features,
                           "out_features": self.current_level}
        elif self.name == 'conv':
            self.params = {'in_channels': input_shape[0]}
            kernel_max = input_shape[1]
            out_range = modulesConfig['conv']['out_channels']
            self.current_level = get_level(out_range)
            self.params['out_channels'] = self.current_level
            kernel_populations = modulesConfig['conv']['kernel_size']
            feasible_kernel_populations = []
            for i in kernel_populations:
                if i <= kernel_max:
                    feasible_kernel_populations.append(i)
            self.params['kernel_size'] = random.sample(feasible_kernel_populations, 1)[0]
            self.params['stride'] = random.sample(modulesConfig['conv']['stride'], 1)[0]
            self.params['padding'] = random.sample(modulesConfig['conv']['padding'], 1)[0]
        elif self.name == 'rnn':
            self.params = dict()
            out_range = modulesConfig['rnn']['hidden_size']
            self.current_level = get_level(out_range)
            self.params['input_size'] = input_shape[0]
            self.params['hidden_size'] = self.current_level
        elif self.name == 'lstm':
            self.params = dict()
            out_range = modulesConfig['lstm']['hidden_size']
            self.current_level = get_level(out_range)
            self.params['input_size'] = input_shape[0]
            self.params['hidden_size'] = self.current_level
        elif self.name == 'maxPool':
            self.params = {'kernel_size': random.sample(modulesConfig['maxPool']['kernel_size'], 1)[0]}
        elif self.name == 'avgPool':
            self.params = {'kernel_size': random.sample(modulesConfig['avgPool']['kernel_size'], 1)[0]}
        elif self.name == 'dropout':
            prob_range = modulesConfig['dropout']['prob_range']
            prob = np.random.uniform(prob_range[0], prob_range[1])
            self.params = {'prob': prob}
        elif self.name == 'bn':
            if len(input_shape) == 1:
                self.params = {'hn_feature': input_shape}
            else:
                self.params = {'hn_feature': input_shape[0]}
        else:
            print(f'not implemented:{self.name}')
            raise ValueError(f'no such module: {self.name}')

        # get output shape
        m = self.get_module_instance()
        input_data = torch.zeros([1, *input_shape])
        output = m(input_data)
        self.output_shape = output.shape[1:]
        self.input_shape = tuple(self.input_shape)
        self.output_shape = tuple(self.output_shape)
        if self.output_shape[1] == 0:
            raise ValueError(f'should not pooling to 0 hn_feature: output shape {self.output_shape}')

    @staticmethod
    def identity_module(name, input_shape: tuple):
        """
        generate an identity mapping module
        :rtype: NAS_Module
        :param name: type of identity module:[dense, conv, rnn]
        :return: identity module (of class Modules)
        """
        if type(name) != str:
            name = NASConfig['editable'][name]
        module = NAS_Module(name, input_shape)

        if name == 'dense':
            dense = nn.Linear(input_shape[-1], input_shape[-1], bias=True)
            dense.weight = nn.Parameter(torch.eye(input_shape[-1]))
            dense.bias = nn.Parameter(torch.zeros(input_shape[-1]))
            module.current_level = input_shape[1]
            module.output_shape = input_shape
            module.params = {'in_features': input_shape[1], 'out_features': input_shape[1]}
            module._module_instance = dense
            return module
        elif name == 'conv':
            out_channel = input_shape[0]
            in_channel = input_shape[0]
            kernel_size = 3
            module.current_level = input_shape[0]
            module.output_shape = input_shape
            module.params = {
                "in_channels": in_channel,
                "out_channels": out_channel,
                "kernel_size": kernel_size,
                "stride": 1,
                "padding": 1,
            }
            conv = nn.Conv1d(in_channels=module.params['in_channels'],
                             out_channels=module.params['out_channels'],
                             kernel_size=module.params['kernel_size'],
                             stride=module.params['stride'],
                             padding=module.params['padding'])
            weight = torch.zeros((out_channel, in_channel, kernel_size))
            for i in range(in_channel):
                weight[i, i, 1] = 1
            conv.weight = nn.Parameter(weight)
            bias = torch.zeros(out_channel)
            conv.bias = nn.Parameter(bias)
            module._module_instance = conv
            return module
        elif name == 'rnn':
            hidden_size = input_shape[0]
            input_size = input_shape[0]
            module.params = {
                "hidden_size": hidden_size,
                "input_size": input_size,
            }
            module.current_level = hidden_size
            module.output_shape = input_shape
            rnn = NAS_RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                nonlinearity='relu',
                batch_first=True,
            )
            bias_ih_l0 = torch.zeros(hidden_size)
            rnn.rnn_unit.bias_ih_l0 = nn.Parameter(bias_ih_l0,True)
            bias_hh_l0 = torch.zeros(hidden_size)
            rnn.rnn_unit.bias_hh_l0 = nn.Parameter(bias_hh_l0,True)
            weight_ih_l0 = torch.eye(hidden_size)
            rnn.rnn_unit.weight_ih_l0 = nn.Parameter(weight_ih_l0,True)
            weight_hh_l0 = torch.zeros((hidden_size, hidden_size))
            rnn.rnn_unit.weight_hh_l0 = nn.Parameter(weight_hh_l0,True)
            module._module_instance = rnn
            return module
        elif name == 'lstm':
            hidden_size = input_shape[0]
            input_size = input_shape[0]
            module.params = {
                "hidden_size": hidden_size,
                "input_size": input_size,
            }
            module.current_level = hidden_size
            module.output_shape = input_shape
            identity_lstm = NAS_SLSTM.generate_identity(input_size)
            module._module_instance = identity_lstm
            return module
        else:
            raise ValueError(f'no such module :{name}')


    def get_module_instance(self):
        """
        generate a model instance once and use it for the rest of procedure
        :return:
        """
        if self._module_instance is not None:
            return self._module_instance

        if self.params is None and self.output_shape is None:
            raise ValueError("parameter must be initialized before generate module instance!!!")
        if self.name == "dense":
            self._module_instance = nn.Linear(self.params['in_features'], self.params['out_features'], bias=True)
        elif self.name == 'conv':
            self._module_instance = nn.Conv1d(
                in_channels=self.params['in_channels'],
                out_channels=self.params['out_channels'],
                kernel_size=self.params['kernel_size'],
                stride=self.params['stride'],
                padding=self.params['padding'])
        elif self.name == 'rnn':
            self._module_instance = NAS_RNN(
                input_size=self.params['input_size'],
                hidden_size=self.params['hidden_size'],
                nonlinearity='relu',
                batch_first=True,
            )
        elif self.name == 'lstm':
            self._module_instance = NAS_SLSTM(
                input_size=self.params['input_size'],
                hidden_size=self.params['hidden_size'],
            )
        elif self.name == 'maxPool':
            self._module_instance = nn.MaxPool1d(self.params['kernel_size'])
        elif self.name == 'avgPool':
            self._module_instance = nn.MaxPool1d(self.params['kernel_size'])
        elif self.name == 'dropout':
            self._module_instance = nn.Dropout(self.params['prob'])
        elif self.name == 'bn':
            self._module_instance = nn.BatchNorm1d(self.params['hn_feature'])
        else:
            print(f'not implemented:{self.name}')
            raise ValueError(f'no such module: {self.name}')
        return self._module_instance

    @property
    def token(self):
        token = None
        if self.name == 'conv':
            token = 'conv-%d-%d' % (self.params['out_channels'], self.params['kernel_size'])
        elif self.name == 'dense':
            token = 'dense-%d' % (self.params['out_features'])
        elif self.name == 'rnn':
            token = 'rnn-%d' % (self.params['hidden_size'])
        elif self.name == 'lstm':
            token = 'lstm-%d' % (self.params['hidden_size'])
        elif self.name == 'maxPool':
            token = 'maxPool-%d' % (self.params['kernel_size'])
        elif self.name == 'avgPool':
            token = 'avgPool-%d' % (self.params['kernel_size'])
        return token

    def __str__(self):
        return f'{self.name}-{self.input_shape}-{self.output_shape}-{self.params}'

    def perform_wider_transformation_current(self):
        """
        generate a new wider module by the wider FPT(function preserving transformation)
        :return: mapping_g, scale_g
        """
        next_level = self.next_level
        if self.name == "dense":
            new_module_instance = nn.Linear(self.params['in_features'], next_level)
            # keep previous parameters
            mapping_g = self.widen_sample_fn(self.current_level, next_level)
            scale_g = [1 / mapping_g.count(i) for i in mapping_g]
            new_module_instance.bias = torch.nn.Parameter(self._module_instance.bias[mapping_g], requires_grad=True)
            new_module_instance.weight = torch.nn.Parameter(self._module_instance.weight[mapping_g], requires_grad=True)
            self.current_level = next_level
            self._module_instance = new_module_instance
            self.output_shape = (self.output_shape[0], self.current_level)
            self.params['out_features'] = self.current_level
        elif self.name == 'conv':
            new_module_instance = nn.Conv1d(in_channels=self.params['in_channels'],
                                            out_channels=self.next_level,
                                            kernel_size=self.params['kernel_size'],
                                            stride=self.params['stride'], padding=self.params['padding'])
            # keep previous parameters
            mapping_g = self.widen_sample_fn(self.current_level, next_level)
            scale_g = [1 / mapping_g.count(i) for i in mapping_g]
            new_module_instance.bias = torch.nn.Parameter(self._module_instance.bias[mapping_g], requires_grad=True)
            new_module_instance.weight = torch.nn.Parameter(self._module_instance.weight[mapping_g], requires_grad=True)
            self.current_level = next_level
            self._module_instance = new_module_instance
            self.output_shape = (self.current_level, self.output_shape[1])
            self.params['out_channels'] = self.current_level
        elif self.name =='rnn':
            new_module_instance = NAS_RNN(
                input_size=self.params['input_size'],
                hidden_size=next_level,
                nonlinearity='relu',
                batch_first=True,
            )
            # keep previous parameters
            mapping_g = self.widen_sample_fn(self.current_level, next_level)
            scale_g = [1 / mapping_g.count(i) for i in mapping_g]
            scale_g = torch.tensor(scale_g)
            rnn = new_module_instance.rnn_unit
            rnn.bias_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_ih_l0[mapping_g] ,True)
            rnn.bias_hh_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_hh_l0[mapping_g] ,True)
            rnn.weight_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[mapping_g] ,True)
            rnn.weight_hh_l0 = nn.Parameter((self._module_instance.rnn_unit.weight_hh_l0[:,mapping_g] * scale_g)[mapping_g],True)
            self.current_level = next_level
            self._module_instance = new_module_instance
            self.output_shape = (self.current_level, self.output_shape[1])
            self.params['hidden_size'] = self.current_level
        elif self.name =='lstm':
            new_module_instance = NAS_SLSTM(self.params['input_size'], next_level)
            # keep previous parameters
            mapping_g = self.widen_sample_fn(self.current_level, next_level)
            scale_g = [1 / mapping_g.count(i) for i in mapping_g]
            scale_g = torch.tensor(scale_g)
            lstm = new_module_instance
            lstm.Wf = nn.Parameter(self._module_instance.Wf[:, mapping_g])
            lstm.Wi = nn.Parameter(self._module_instance.Wi[:, mapping_g])
            lstm.Wo = nn.Parameter(self._module_instance.Wo[:, mapping_g])
            lstm.Wc = nn.Parameter(self._module_instance.Wc[:, mapping_g])
            lstm.bf = nn.Parameter(self._module_instance.bf[mapping_g])
            lstm.bi = nn.Parameter(self._module_instance.bi[mapping_g])
            lstm.bo = nn.Parameter(self._module_instance.bo[mapping_g])
            lstm.bc = nn.Parameter(self._module_instance.bc[mapping_g])
            lstm.Uf = nn.Parameter((self._module_instance.Uf.T[:, mapping_g] * scale_g)[mapping_g].T)
            lstm.Ui = nn.Parameter((self._module_instance.Ui.T[:, mapping_g] * scale_g)[mapping_g].T)
            lstm.Uo = nn.Parameter((self._module_instance.Uo.T[:, mapping_g] * scale_g)[mapping_g].T)
            lstm.Uc = nn.Parameter((self._module_instance.Uc.T[:, mapping_g] * scale_g)[mapping_g].T)
            self.current_level = next_level
            self._module_instance = new_module_instance
            self.output_shape = (self.current_level, self.output_shape[1])
            self.params['hidden_size'] = self.current_level
        return mapping_g, scale_g

    def perform_wider_transformation_next(self, mapping_g: list, scale_g: list):
        """
        generate a new wider module by the wider FPT(function preserving transformation)
        :return: module of next level
        """
        next_level = len(mapping_g)
        scale_g = torch.tensor(scale_g)
        if self.name == "dense":
            new_module_instance = nn.Linear(next_level, self.params['out_features'])
            # keep previous parameters
            new_module_instance.weight = torch.nn.Parameter(
                self._module_instance.weight[:, mapping_g] * scale_g.unsqueeze(0), requires_grad=True)
            self._module_instance = new_module_instance
            self.input_shape = (self.input_shape[0], next_level)
            self.params['in_features'] = next_level
        elif self.name == 'conv':
            new_module_instance = nn.Conv1d(in_channels=next_level,
                                            out_channels=self.params['out_channels'],
                                            kernel_size=self.params['kernel_size'],
                                            stride=self.params['stride'], padding=self.params['padding'])
            new_module_instance.weight = \
                torch.nn.Parameter(self._module_instance.weight[:, mapping_g] *
                                   scale_g.unsqueeze(0).unsqueeze(2),
                                   requires_grad=True)
            self._module_instance = new_module_instance
            self.input_shape = (next_level, self.input_shape[1])
            self.params['in_channels'] = next_level
        elif self.name == 'rnn':
            new_module_instance = NAS_RNN(
                input_size=next_level,
                hidden_size=self.params['hidden_size'],
                nonlinearity='relu',
                batch_first=True,
            )
            rnn = new_module_instance.rnn_unit
            rnn.bias_ih_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_ih_l0,True)
            rnn.bias_hh_l0 = nn.Parameter(self._module_instance.rnn_unit.bias_hh_l0,True)
            rnn.weight_ih_l0 = \
                torch.nn.Parameter(self._module_instance.rnn_unit.weight_ih_l0[:, mapping_g] *
                                   scale_g.unsqueeze(0),
                                   requires_grad=True)
            rnn.weight_hh_l0 = \
                torch.nn.Parameter(self._module_instance.rnn_unit.weight_hh_l0,
                                   requires_grad=True)
            self._module_instance = new_module_instance
            self.input_shape = (next_level, self.input_shape[1])
            self.params['input_size'] = next_level
        elif self.name == 'lstm':
            new_module_instance = NAS_SLSTM(next_level, self.params['hidden_size'])
            lstm = new_module_instance
            lstm.Wf = nn.Parameter(self._module_instance.Wf[mapping_g] * scale_g.unsqueeze(1))
            lstm.Wi = nn.Parameter(self._module_instance.Wi[mapping_g] * scale_g.unsqueeze(1))
            lstm.Wo = nn.Parameter(self._module_instance.Wo[mapping_g] * scale_g.unsqueeze(1))
            lstm.Wc = nn.Parameter(self._module_instance.Wc[mapping_g] * scale_g.unsqueeze(1))
            lstm.bf = nn.Parameter(self._module_instance.bf)
            lstm.bi = nn.Parameter(self._module_instance.bi)
            lstm.bo = nn.Parameter(self._module_instance.bo)
            lstm.bc = nn.Parameter(self._module_instance.bc)
            lstm.Uf = nn.Parameter(self._module_instance.Uf)
            lstm.Ui = nn.Parameter(self._module_instance.Ui)
            lstm.Uo = nn.Parameter(self._module_instance.Uo)
            lstm.Uc = nn.Parameter(self._module_instance.Uc)
            self._module_instance = new_module_instance
            self.input_shape = (next_level, self.input_shape[1])
            self.params['input_size'] = next_level

    def default_sample_strategy(self, original_size, new_size):
        seq = list(range(original_size))
        num_to_sample = new_size - original_size
        while num_to_sample != 0:
            sample_number = min(original_size, num_to_sample)
            seq += random.sample(seq, sample_number)
            num_to_sample = num_to_sample - sample_number
        return seq


class NAS_RNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # kwargs['bias'] = False
        self.rnn_unit = nn.RNN(*args, **kwargs)

    def forward(self, x: Tensor):
        x = x.permute(0, 2, 1)
        output, hn = self.rnn_unit(x)
        y = output
        y = y.permute(0, 2, 1)
        return y


class NAS_SLSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wf = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wi = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wc = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Wo = nn.Parameter(torch.empty(input_size, hidden_size))
        self.Uf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Ui = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Uc = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Uo = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        self.bc = nn.Parameter(torch.zeros(hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.Wf, a=math.sqrt(5))
        init.kaiming_uniform_(self.Wi, a=math.sqrt(5))
        init.kaiming_uniform_(self.Wc, a=math.sqrt(5))
        init.kaiming_uniform_(self.Wo, a=math.sqrt(5))
        init.kaiming_uniform_(self.Uf, a=math.sqrt(5))
        init.kaiming_uniform_(self.Ui, a=math.sqrt(5))
        init.kaiming_uniform_(self.Uc, a=math.sqrt(5))
        init.kaiming_uniform_(self.Uo, a=math.sqrt(5))


    def forward_step(self, x, h, c):
        """
        do one step LSTM forward calculation
        @param x: input shape [batch, feature]
        """
        ft = torch.sigmoid(torch.matmul(x, self.Wf) + torch.matmul(h, self.Uf) + self.bf)
        it = torch.sigmoid(torch.matmul(x, self.Wi) + torch.matmul(h, self.Ui) + self.bi)
        ot = torch.matmul(x, self.Wo) + torch.matmul(h, self.Uo) + self.bo
        c_ = torch.tanh(torch.matmul(x, self.Wc) + torch.matmul(h, self.Uc) + self.bc)
        c = ft * c + it * c_
        h = ot * torch.tanh(c) + ot
        return h, c

    def forward(self, x):
        '''
        forward propagation
        @param x:  input of size [batch, feature, seqlen]
        @return: hidden state (batch, feature,seqlen)
        '''
        h = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
        c = torch.zeros((x.shape[0], self.hidden_size), device=x.device)
        output = []
        for i in range(x.shape[2]):
            h, c = self.forward_step(x[:,:,i], h, c)
            output.append((h,c))
        output = [torch.stack([i[0] for i in output]),torch.stack([i[1] for i in output])]
        return output[0].permute(1, 2, 0)

    @staticmethod
    def generate_identity(input_size):
        lstm = NAS_SLSTM(input_size, input_size)
        lstm.Wo = nn.Parameter(torch.eye(input_size))
        lstm.Wf = nn.Parameter(torch.zeros_like(lstm.Wf))
        lstm.Wi = nn.Parameter(torch.zeros_like(lstm.Wi))
        lstm.Wc = nn.Parameter(torch.zeros_like(lstm.Wc))
        lstm.Uf = nn.Parameter(torch.zeros_like(lstm.Uf))
        lstm.Uc = nn.Parameter(torch.zeros_like(lstm.Uc))
        lstm.Ui = nn.Parameter(torch.zeros_like(lstm.Ui))
        lstm.Uo = nn.Parameter(torch.zeros_like(lstm.Uo))
        lstm.bf = nn.Parameter(torch.zeros_like(lstm.bf))
        lstm.bc = nn.Parameter(torch.zeros_like(lstm.bc))
        lstm.bi = nn.Parameter(torch.zeros_like(lstm.bi))
        lstm.bo = nn.Parameter(torch.zeros_like(lstm.bo))
        return lstm



def generate_from_skeleton(skeleton: list, input_shape):
    modules = []
    for name in skeleton + ['dense']:
        module = NAS_Module(name, input_shape)
        try:
            module.small_init_param(input_shape)
            input_shape = module.output_shape
            modules.append(module)
        except Exception as e:
            pass
            # logger.warn(f'too much pooling makes empty result, skip module:{module}')

    return modules
