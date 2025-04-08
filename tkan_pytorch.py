"""
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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TKANLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, sub_activations, use_bias=True):
        super(TKANLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.sub_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=use_bias) for _ in sub_activations
        ])
        self.activations = []
        for act in sub_activations:
            if isinstance(act, str) and hasattr(F, act):
                self.activations.append(getattr(F, act))
            else:
                self.activations.append(lambda x: x)  # Identity for unsupported/None

    def forward(self, x):
        outputs = [act(layer(x)) for layer, act in zip(self.sub_layers, self.activations)]
        return sum(outputs)  # Sum instead of average for KAN-like behavior

class TKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, sub_kan_configs=['relu', 'relu', 'relu'], return_sequences=False, use_bias=True):
        super(TKAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        # Combine input and hidden state
        self.tkan_layer = TKANLayer(input_dim + hidden_dim, hidden_dim, sub_kan_configs, use_bias)
        self.initial_hidden = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x):
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {D}")
        hidden = self.initial_hidden.expand(B, -1)  # [B, hidden_dim]
        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # [B, input_dim]
            combined = torch.cat([xt, hidden], dim=-1)  # [B, input_dim + hidden_dim]
            hidden = self.tkan_layer(combined)  # [B, hidden_dim]
            outputs.append(hidden)
        outputs = torch.stack(outputs, dim=1)  # [B, T, hidden_dim]
        return outputs if self.return_sequences else outputs[:, -1, :]  # [B, T, hidden_dim] or [B, hidden_dim]
