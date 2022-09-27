import torch
import torch.nn as nn

from collections import OrderedDict, namedtuple
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return  input_ / (1+torch.abs(input_))

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (torch.abs(input_) + 1.0) ** 2
    
fast_sigmoid = FastSigmoid.apply

class MetaLIF(MetaModule):
    NeuronState = namedtuple('NeuronState', ['V', 'S'])
    def __init__(self, in_features, out_features, decay):
        super(MetaLIF, self).__init__()
        self.layer = MetaLinear(in_features, out_features)
        self.register_buffer('decay', torch.exp(-1./torch.tensor(decay)))

        self.state = None
    
    def reset(self):
        self.state = None

    def forward(self, inputs, params=None):
        I = self.layer(inputs, params=self.get_subdict(params, 'layer'))

        if self.state is None:
            self.state = self.NeuronState(V = torch.zeros_like(I),
                                          S = torch.zeros_like(I))

        V = self.state.V
        S = self.state.S

        V = self.decay*V + I - S.detach()
        S = fast_sigmoid(V)

        self.state = self.NeuronState(V=V, S=S)

        return S

class MetaSNNModel(MetaModule):
    def __init__(self, in_features, out_features, hidden_sizes, decay):
        super(MetaSNNModel, self).__init__()

        layer_sizes = [in_features] + hidden_sizes
        self.LIFlayers = MetaSequential(OrderedDict([('LIFlayer{0}'.format(i + 1),
            MetaLIF(hidden_size, layer_sizes[i+1], decay)
            ) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLIF(hidden_sizes[-1], out_features, decay)

    def forward(self, inputs, params=None):
        features = self.LIFlayers(inputs.view(inputs.shape[0], -1), params=self.get_subdict(params, 'LIFlayers'))
        logits = self.classifier(features)
        return logits

    def reset(self):
        for layer in self.LIFlayers:
            layer.reset()
        self.classifier.reset()

def ModelSNN(in_features, out_features, hidden_sizes=[400, 200, 128], decay=20.):
    return MetaSNNModel(in_features, out_features, hidden_sizes=hidden_sizes, decay=decay)