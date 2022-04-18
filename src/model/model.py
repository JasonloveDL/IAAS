import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn
from torch.utils.data.dataloader import *
from torch.utils.data.dataset import TensorDataset
from torchinfo import torchinfo

import utils
from utils.sql_connector import *
from .module import NAS_Module

logger = get_logger('NasModel')

total_model_count = None
activate = torch.nn.ReLU


def reset_model_count():
    global total_model_count
    total_model_count = None


class ModelConfig:
    _index: int

    def __init__(self, modules: list, feature_shape, target_shape):
        self.target_shape = target_shape
        self.feature_shape = feature_shape
        self.modules = modules
        self.tail_layers = []

    @property
    def insert_length(self):
        return len(self.modules) + 1

    @property
    def widenable_list(self):
        widenable = [m.widenable for m in self.modules]
        for i in range(len(self.modules) - 1):
            # Widening is allowed for the previous one of the continuously same widening modules
            if widenable[i] and \
                    self.modules[i].name == self.modules[i + 1].name \
                    and not self.modules[i].is_max_level:
                widenable[i] = True
            else:
                widenable[i] = False
        widenable[-1] = False  # The last layer is not allowed to be widened
        return widenable

    @property
    def can_widen(self):
        can_widen = False
        for i in self.widenable_list:
            if i:
                can_widen = True
        return can_widen

    def generate_model(self):
        """
        generate a new NasModel from scratch, all NasModel trainable parameters is randomly generated
        :return: torch.NAS_Module
        """
        module_instances = []
        for m in self.modules:
            module_instances.append(m.get_module_instance())
            module_instances.append(activate())
        output_shape = self.modules[-1].output_shape
        if len(output_shape) > 1:
            s = 1
            for i in output_shape:
                s *= i
            output_shape = s
            self.tail_layers.append(torch.nn.Flatten())
        self.tail_layers.append(torch.nn.Linear(output_shape, self.target_shape))
        module_instances = [*module_instances, *self.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        return NasModel(model_instance, self)

    @property
    def token_list(self):
        token_list = []
        for m in self.modules:
            assert type(m) == NAS_Module
            token_list.append(m.token)
        return token_list

    def __str__(self):
        return str.join('->', self.token_list)

    def set_index(self, index):
        self._index = index

    def get_index(self):
        return self._index


class NasModel:
    def __init__(self, model_instance: torch.nn.Module, model_config: ModelConfig, prev_index=-1):
        self.model_config = model_config
        self.model_instance = model_instance
        self.train_times = 0  # record total train times
        self.loss_list = []
        self.test_loss_best = None
        self.test_loss_best_iteration = 0
        self.test_loss = None
        self.next_save = False
        self.prev_index = prev_index
        # save global NasModel information
        self.update_global_information()
        self.transformation_record = pd.DataFrame({'prev': -1, 'current': self.index, 'train_times': 0}, index=[0])
        self.optimizer = None
        self.activate = activate

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return hash(self.index)

    def to_cuda(self):
        self.model_instance.cuda()

    def to_cpu(self):
        self.model_instance.cpu()

    def add_transformation_record(self, prev, current, train_times, loss_list):
        self.transformation_record = self.transformation_record.append(
            {'prev': prev, 'current': current, 'train_times': train_times}, ignore_index=True)
        self.loss_list = copy.deepcopy(loss_list)

    def update_global_information(self):
        global total_model_count
        if total_model_count is None:
            total_model_count = get_total_model_count()
        try:
            total_model_count += 1
            self.index = total_model_count
            insert_new_model_config(self.index, str(self.model_config), self.prev_index)
        except:
            id, structure, train_time, loss, prev_index = get_prev_record(str(self.model_config))
            self.index = id
            self.prev_index = prev_index
            # if prev_index == -1:
            #     self.train_times = train_time
            #     self.test_loss_best = loss
            #     model_dir = os.path.join(NASConfig['OUT_DIR'], f'{self.index}')
            #     model_path = os.path.join(model_dir, 'NasModel.pth')
            #     if os.path.exists(model_path):
            #         self.model_instance = torch.load(model_path)
            #         logger.critical(f'load prev model {id}, train {train_time} loss {loss}')
        logger.info(f'create model {self.index} from {self.prev_index}, structure: {self.model_config}')

    def __call__(self, x):
        return self.model_instance(x)

    def __str__(self):
        return str(self.index) + " " + str(self.model_config)

    def add_noise(self, std=1):
        """
        add small white noise to net parameters
        @return:
        """
        parameters = list(self.model_instance.parameters())
        for parameter in parameters:
            parameter.data += torch.randn_like(parameter.data) * std

    def train(self, X_train, y_train):
        """
        train NasModel directly(without batch), all feature make forward computation at once
        :param X_train: train hn_feature
        :param y_train: train targets
        :return: None
        """
        optimizer = self._get_optimizer()
        loss_fn = self._get_loss_function()
        dataloader = DataLoader(TensorDataset(X_train, y_train),
                                NASConfig['BATCH_SIZE'],
                                shuffle=True)
        for i in range(NASConfig['IterationEachTime']):
            # Compute prediction and loss
            st = time.time()
            loss_list = []
            for step, (batch_x, batch_y) in enumerate(dataloader):
                pred = self.model_instance(batch_x)
                pred = pred.view(-1)
                loss = loss_fn(pred, batch_y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            # show train process
            # self.show_train_process(pred, batch_y)
            self.train_times += 1
            self.loss_list.append(np.mean(loss_list))
            if self.train_times % NASConfig['MonitorIterations'] == 0:
                logger.info(f'NasModel {self.index} '
                            f'train {self.train_times} times '
                            f'loss {loss.item()} batch: {len(loss_list)} * {NASConfig["BATCH_SIZE"]} '
                            f'{NASConfig["IterationEachTime"]} epoch time: {(time.time() - st) * NASConfig["IterationEachTime"]} sec.')

    def test(self, X_test, y_test):
        with torch.no_grad():
            loss_fn = self.rmse
            pred = self.model_instance(X_test).view(-1)
            loss = loss_fn(pred, y_test)
            self.test_loss = loss.item()
            self.next_save = False
            if np.isnan(self.test_loss):
                self.test_loss = 1e5
            if self.test_loss_best is None:
                self.test_loss_best = self.test_loss
                self.next_save = True
            if self.test_loss_best > self.test_loss:
                self.test_loss_best = self.test_loss
                self.next_save = True
            if self.next_save:
                self.test_loss_best_iteration = len(self.loss_list)
            return self.test_loss

    def perform_wider_transformation(self, wider_index):
        """
        generate a new NasModel by widen the specific layer
        the widen method follows Net2Net(arXiv:1511.05641)
        :param wider_index: index of layer to widen
        :return: new NasModel with the widen layer
        """
        self.to_cpu()
        model_config = copy.deepcopy(self.model_config)
        module_instances = []
        i = 0
        while True:
            if i == len(model_config.modules):
                break
            if i == wider_index:
                # wider the specific network
                module = model_config.modules[i]
                mapping_g, scale_g = module.perform_wider_transformation_current()
                module_instances.append(model_config.modules[i].get_module_instance())
                module_instances.append(self.activate())
                i += 1
                # modify next module to match current module
                module = model_config.modules[i]
                module.perform_wider_transformation_next(mapping_g, scale_g)
                module_instances.append(model_config.modules[i].get_module_instance())
                module_instances.append(self.activate())

            else:
                # inherent from father network
                module_instances.append(model_config.modules[i].get_module_instance())
                module_instances.append(self.activate())
            i += 1
        module_instances = [*module_instances, *model_config.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        m = NasModel(model_instance, model_config, self.index)
        m.transformation_record = self.transformation_record.copy()
        m.add_transformation_record(self.index, m.index, self.train_times, self.loss_list)
        return m

    def perform_deeper_transformation(self, deeper_action):
        """
        generate a new NasModel by insert a new layer
        the deeper method follows Net2Net(arXiv:1511.05641)
        :param deeper_action: (insert layer type, insert layer place)
        :return: new NasModel with deeper layers
        """
        self.to_cpu()
        insert_type, insert_index = deeper_action
        model_config = copy.deepcopy(self.model_config)
        module_instances = []
        module_length = len(model_config.modules)
        identity_module = None
        for i in range(module_length):
            if i == insert_index:
                # insert layer inside the network
                input_shape = self.model_config.modules[i].input_shape
                identity_module = NAS_Module.identity_module(insert_type, input_shape)
                module_instances.append(identity_module.get_module_instance())
                module_instances.append(self.activate())

            module_instances.append(model_config.modules[i].get_module_instance())
            module_instances.append(self.activate())

        # insert in tail
        if insert_index == module_length:
            input_shape = self.model_config.modules[-1].output_shape
            identity_module = NAS_Module.identity_module(insert_type, input_shape)
            module_instances.append(identity_module.get_module_instance())
            module_instances.append(self.activate())
        model_config.modules.insert(insert_index, identity_module)

        module_instances = [*module_instances, *model_config.tail_layers]
        model_instance = torch.nn.Sequential(*module_instances)
        m = NasModel(model_instance, model_config, prev_index=self.index)
        m.transformation_record = self.transformation_record.copy()
        m.add_transformation_record(self.index, m.index, self.train_times, self.loss_list)
        return m

    def save_pred_result(self, X_test, y_test):
        model_dir = os.path.join(NASConfig['OUT_DIR'], f'{self.index}')
        pred_result_path = os.path.join(model_dir, 'pred.csv')
        pred_figure_path = os.path.join(model_dir, 'pred.png')
        detail_pred_figure_path = os.path.join(model_dir, 'detail_pred.png')
        best_pred_figure_path = os.path.join(model_dir, 'best_pred.png')
        best_detail_pred_figure_path = os.path.join(model_dir, 'best_detail_pred.png')
        with torch.no_grad():
            pred = self.model_instance(X_test)
        y_test = y_test.cpu()
        pred = pred.cpu()
        plt.plot(y_test, label='y')
        plt.plot(pred, label='pred')
        plt.legend(loc='upper left')
        plt.savefig(pred_figure_path)
        if self.next_save:
            plt.savefig(best_pred_figure_path)
        plt.clf()
        plt.plot(y_test[:200], label='y')
        plt.plot(pred[:200], label='pred')
        plt.legend(loc='upper left')
        plt.savefig(detail_pred_figure_path)
        if self.next_save:
            plt.savefig(best_detail_pred_figure_path)
            # save prediction result
            pd.DataFrame({'truth': y_test.view(-1), 'pred': pred.view(-1)}).to_csv(pred_result_path)
        plt.clf()
        plt.close()

    def save_model(self):
        model_dir = os.path.join(NASConfig['OUT_DIR'], f'{self.index}')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'NasModel.pth')
        loss_figure_path = os.path.join(model_dir, 'loss.png')
        model_str_path = os.path.join(model_dir, 'NasModel.txt')
        model_transformation_path = os.path.join(model_dir, 'transformation.csv')
        loss_path = os.path.join(model_dir, 'loss.csv')
        if self.next_save:
            torch.save(self.model_instance, model_path)
        plt.plot(self.loss_list)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        # x0, y0 = self.test_loss_best_iteration, self.loss_list[self.train_times - 1]
        # plt.plot(x0, y0, 'om')
        # plt.annotate('save model', xy=(x0, y0), xytext=(x0, y0))
        plt.savefig(loss_figure_path)
        plt.close()
        with open(model_str_path, 'w') as f:
            f.write(str(self.model_config))
        pd.DataFrame(self.loss_list).to_csv(loss_path)
        self.transformation_record.to_csv(model_transformation_path)

        insert_new_train_result(self.index, self.train_times, self.test_loss_best)

    def _get_loss_function(self):

        return self.rmse

    def _get_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        self.optimizer = torch.optim.Adam(self.model_instance.parameters())
        return self.optimizer

    @staticmethod
    def mape(pred, truth):
        return torch.mean(torch.abs((pred - truth) / (truth + 1e-5)))

    @staticmethod
    def rmse(pred, truth):
        import torch.nn.functional as F
        return torch.sqrt(F.mse_loss(pred, truth))

    @staticmethod
    def get_param_number(net):
        return torchinfo.summary(net, verbose=False).total_params

    @staticmethod
    def show_train_process(pred, batch_y):
        plt.clf()
        plt.ion()
        plt.plot(pred.detach().cpu().numpy(), label='pred')
        plt.plot(batch_y.detach().cpu().numpy(), label='y')
        plt.legend()
        plt.pause(0.01)


def generate_new_model_config(feature_shape, targe_shape, skeleton=None) -> ModelConfig:
    """
    generate random NasModel from scratch
    :return: ModelConfig of new NasModel
    """
    # generate NasModel configuration
    from model import generate_from_skeleton
    if skeleton is None:
        maxLayers = utils.NASConfig['MaxInitLayers']
        layers = np.random.randint(1, maxLayers + 1)
        skeleton = [j[0] for j in [random.sample(utils.modulesList, 1) for i in range(layers)]]
    modules = generate_from_skeleton(skeleton, feature_shape)
    # generate NasModel train operations
    config = ModelConfig(modules, feature_shape, targe_shape)
    return config
