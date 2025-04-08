import torch
import torch.nn as nn
import torch.nn.functional as F

class TKANLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, sub_activations, use_bias=True):
        super(TKANLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.sub_layers = nn.ModuleList()

        for act in sub_activations:
            layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
            self.sub_layers.append(layer)

        self.activations = []
        for act in sub_activations:
            if isinstance(act, str):
                if hasattr(F, act):
                    self.activations.append(getattr(F, act))
                else:
                    raise ValueError(f"Unsupported activation: {act}")
            else:
                self.activations.append(lambda x: x)

    def forward(self, x):
        outputs = []
        for layer, activation in zip(self.sub_layers, self.activations):
            outputs.append(activation(layer(x)))
        return sum(outputs) / len(outputs)

class TKAN(nn.Module):
    def __init__(self, input_dim, sub_kan_configs=['relu', 'relu', 'relu'], return_sequences=False, use_bias=True):
        super(TKAN, self).__init__()
        self.return_sequences = return_sequences
        self.tkan_layer = TKANLayer(input_dim, input_dim, sub_kan_configs, use_bias)

    def forward(self, x):
        B, T, D = x.shape
        outputs = []
        for t in range(T):
            xt = x[:, t, :]
            out = self.tkan_layer(xt)
            outputs.append(out)
        outputs = torch.stack(outputs, dim=1)
        if self.return_sequences:
            return outputs
        else:
            return outputs[:, -1, :]
