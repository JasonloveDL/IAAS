import torch
from torch.distributions import Categorical
from torch.nn import *

from utils.config_loader import modulesConfig, NASConfig

_vocabulary = None  # save vocabulary for the whole NAS architecture, used for embedding


class Vocabulary:
    """
    store embedding code of each NAS module
    """
    def __init__(self, token_list):
        token_list = token_list
        self.vocab = {}
        for idx, token in enumerate(token_list):
            self.vocab[token] = idx
            self.vocab[idx] = token

    @property
    def size(self):
        return len(self.vocab) // 2

    def get_code(self, token_list):
        return [self.vocab[token] for token in token_list]

    def get_token(self, code_list):
        return [self.vocab[code] for code in code_list]

    def __str__(self):
        return str(self.vocab)


def get_vocabulary():
    global _vocabulary
    if _vocabulary is not None:
        return _vocabulary
    token_expand = 10  # give large enough token space
    token_list = []
    token_list += ['conv-%d-%d' % (f, k) for f in range(1,token_expand * max(modulesConfig['conv']['out_channels']) + 1) for k in modulesConfig['conv']['kernel_size']]
    token_list += ['dense-%d' % i for i in range(1,token_expand * max(modulesConfig['dense']['out_range']) + 1)]
    token_list += ['rnn-%d' % i for i in range(1,token_expand * max(modulesConfig['rnn']['hidden_size']) + 1)]
    _vocabulary = Vocabulary(token_list)
    return _vocabulary


class EncoderNet(Module):
    """
    Encoder network similar to EAS
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_unit = torch.nn.LSTM(input_size, hidden_size, 1, True, True, bidirectional=True)
        self.embedding_unit = torch.nn.Embedding(get_vocabulary().size, input_size)
        if NASConfig['GPU']:
            self.embedding_unit = self.embedding_unit.cuda()
            self.lstm_unit = self.lstm_unit.cuda()


    def embedding(self, token_list):
        vocab = get_vocabulary()
        codes = vocab.get_code(token_list)
        input = torch.tensor(codes)
        if NASConfig['GPU']:
            input = input.cuda()
        out = self.embedding_unit(input)

        return out.unsqueeze(0)

    def forward(self,token_list):
        """
        get encoder output
        :param token_list: list of token: list of str
        :return: output, (h_n, c_n) as described in lstm
        """
        embedding_tensor = self.embedding(token_list)
        output, (h_n, c_n) = self.lstm_unit(embedding_tensor)
        return output, (h_n, c_n)


class WinderActorNet(Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.winderNet = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self,features, editable):
        output = self.winderNet(features.squeeze()).squeeze()
        # filter layers which cannot be widen
        mask = torch.tensor([1 if editable[i] else 0 for i in range(output.shape[0])])
        if NASConfig['GPU']:
            mask = mask.cuda()
        output = output * mask
        output = output / (output.sum() + 1e-9)
        return output

    def get_action(self, features, editable):
        """-ko
        get output of wider actor network
        :param features: hidden states of each network module
        :param editable: indicating if a layer can be widen
        :return: index of layer to be widen
        """
        output_prob = self.forward(features, editable)
        output = Categorical(output_prob).sample()
        return output, output_prob


class DeeperActorNet(Module):
    def __init__(self, input_size: int, max_layers):
        """
        deeper net actor net for make decision on deeper the orignial neural network
        :param input_size: hidden size of encoder network (including bidirectional enlargement of size)
        :param max_layers: maximum layer number of neural network
        """
        super().__init__()
        self.max_layers = max_layers
        self.decision_num = 2
        self.deeperNet = RNN(input_size, input_size,batch_first=True)
        self.insert_type_layer = Sequential(
            Linear(input_size, len(NASConfig['editable'])),
            Sigmoid(),
        )
        self.insert_index_layer = Sequential(
            Linear(input_size, max_layers, True),
            Sigmoid(),
        )

    def get_action(self, hn_feature: torch.Tensor, insert_length:int):
        """
        get output of deeper actor network
        :param hn_feature: last hidden states of encoder network
        :param insert_length: max index of the inserted neural layer
        :return: layer type, insert index, new layer width
        """
        hn_feature = hn_feature.reshape(1, -1)
        hn_feature = torch.stack([hn_feature] * self.decision_num, 1)
        if NASConfig['GPU']:
            hn_feature = hn_feature.cuda()
        output, _ = self.deeperNet(hn_feature)
        insert_type_prob = self.insert_type_layer(output[-1, 0, :])
        insert_type = Categorical(insert_type_prob).sample()
        mask = torch.tensor([1 if i < insert_length else 0 for i in range(self.max_layers)])
        if NASConfig['GPU']:
            mask = mask.cuda()
        insert_index = self.insert_index_layer(output[:, 1,:]) * mask
        insert_index_prob = insert_index/ insert_index.sum()
        insert_index = Categorical(insert_index_prob).sample()
        return (insert_type, insert_index), (insert_type_prob, insert_index_prob)


class SelectorActorNet(Module):
    """
    selector network, output is 3 dimension:
    3 dim:
    (do nothing, wider, deeper)
    """
    def __init__(self, input_size):
        super().__init__()
        self.net = Sequential(
            Linear(input_size,128),
            LeakyReLU(),
            Linear(128, 3),
            Sigmoid(),
        )
        self.no_widen_mask = torch.tensor([1, 0, 1])
        if NASConfig['GPU']:
            self.no_widen_mask = self.no_widen_mask.cuda()

    def forward(self, x):
        x = torch.flatten(x)
        out = self.net(x)
        return out

    def get_action(self, x, can_widen):
        x = torch.flatten(x)
        prob = self.net(x)
        if not can_widen:  # do not make widen action when not layer can widen
            prob = prob * self.no_widen_mask
        prob = prob / prob.sum()
        action = Categorical(prob).sample()
        return action, prob
