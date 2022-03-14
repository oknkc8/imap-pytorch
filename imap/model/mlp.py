import torch
import torch.nn as nn
from ..torchmeta.modules import (MetaModule, MetaSequential)
from ..torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
import pdb

class MLP(nn.Module):
    def __init__(self, input_dimension, output_dimension=4, hidden_features=256, num_hidden_layers=4, last_ac=None):
        super().__init__()
        hidden_dimensions = [hidden_features for _ in range(num_hidden_layers)]
        if len(hidden_dimensions) <= 2:
            self.first_layer = nn.Sequential(*self.make_layers(input_dimension, hidden_dimensions[:1]))
            self.layers = None
        else:
            self.first_layer = nn.Sequential(*self.make_layers(input_dimension, hidden_dimensions[:(num_hidden_layers//2)]))
            input_dimension = input_dimension + hidden_dimensions[(num_hidden_layers//2) - 1]
            models = self.make_layers(input_dimension, hidden_dimensions[(num_hidden_layers//2):])
            self.layers = nn.Sequential(*models)
        # models.append(nn.Linear(hidden_dimensions[-1], output_dimension))
        # models[-1].weight.data[3, :] *= 0.1
        self.last_hidden_dim = hidden_dimensions[-1]
        # self.last_layers = nn.Sequential(*models)
        if last_ac is not None:
            self.last_layer = nn.Sequential(*[nn.Linear(hidden_dimensions[-1], output_dimension), nn.Tanh()])
        else:
            self.last_layer = nn.Linear(hidden_dimensions[-1], output_dimension)

        print(self)

    def forward(self, x):
        before_last_layer = self.first_layer(x)
        if self.layers is not None:
            x = torch.cat([before_last_layer, x], dim=1)
            last_feature = self.layers(x)
            # return self.last_layers(x)
        else:
            last_feature = before_last_layer
        return self.last_layer(last_feature), last_feature

    @staticmethod
    def make_layers(input_dimension, hidden_dimensions):
        models = []
        for hidden_dimension in hidden_dimensions:
            models.append(nn.Linear(input_dimension, hidden_dimension))
            models.append(nn.ReLU())
            input_dimension = hidden_dimension
        return models




class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        # print('input:', input.shape, '/ weight:', weight.shape)
        # print('permuted weight:', weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2).shape)

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        # print('output:', output.shape)
        # print('-'*10)
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        # if outermost_linear:
        #     self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        # else:
        #     self.net.append(MetaSequential(
        #         BatchLinear(hidden_features, out_features), nl
        #     ))

        if outermost_linear:
            self.last_layer = MetaSequential(BatchLinear(hidden_features, out_features))
        else:
            self.last_layer = MetaSequential(BatchLinear(hidden_features, out_features), nl)

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)
            self.last_layer.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        last_feature = self.net(coords, params=get_subdict(params, 'net'))
        output = self.last_layer(last_feature, params=get_subdict(params, 'last_layer'))
        return output, last_feature


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        self.last_hidden_dim = hidden_features
        print(self)

    def forward(self, coords, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output, last_feature = self.net(coords, get_subdict(params, 'net'))
        return output, last_feature

########################
# Initialization methods

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
