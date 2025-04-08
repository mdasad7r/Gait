import torch
import torch.nn as nn
import torch.nn.functional as F

class SplineActivation(nn.Module):
    def __init__(self, input_dim, spline_order=3, grid_size=5):
        super(SplineActivation, self).__init__()
        self.spline_order = spline_order
        self.grid_size = grid_size
        # Grid points (learnable)
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)
        # Coefficients for B-spline basis (learnable)
        self.coeffs = nn.Parameter(torch.randn(input_dim, grid_size + spline_order - 1))

    def _bspline_basis(self, x):
        """Compute B-spline basis functions recursively."""
        B = torch.zeros(x.shape[0], self.grid_size + self.spline_order - 1, device=x.device)
        # Degree 0 basis
        for i in range(self.grid_size - 1):
            B[:, i] = ((self.grid[i] <= x) & (x < self.grid[i + 1])).float()
        B[:, self.grid_size - 1] = (x >= self.grid[-1]).float()

        # Higher-order basis
        for k in range(1, self.spline_order + 1):
            B_prev = B.clone()
            for i in range(self.grid_size + self.spline_order - k - 1):
                left = (x - self.grid[i]) / (self.grid[i + k] - self.grid[i]) * B_prev[:, i]
                right = (self.grid[i + k + 1] - x) / (self.grid[i + k + 1] - self.grid[i + 1]) * B_prev[:, i + 1]
                B[:, i] = left + right
        return B

    def forward(self, x):
        # Apply B-spline basis to each input dimension
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1)  # [B, input_dim, 1]
        basis = self._bspline_basis(x_expanded)  # [B, input_dim, grid_size + spline_order - 1]
        output = torch.einsum('bid,di->bi', basis, self.coeffs)  # [B, input_dim]
        return output

class TKANLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, sub_activations, use_bias=True):
        super(TKANLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.sub_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, bias=use_bias) for _ in sub_activations
        ])
        self.activations = []
        for act in sub_activations:
            if isinstance(act, int):  # Spline order
                self.activations.append(SplineActivation(hidden_dim, spline_order=act))
            elif isinstance(act, str) and hasattr(F, act):  # Standard activation
                self.activations.append(getattr(F, act))
            else:
                self.activations.append(lambda x: x)  # Identity

    def forward(self, x):
        outputs = [act(layer(x)) for layer, act in zip(self.sub_layers, self.activations)]
        return sum(outputs)  # Sum sub-layer outputs as in KAN

class TKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, sub_kan_configs=[3, 2, 1], return_sequences=False, use_bias=True):
        super(TKAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.tkan_layer = TKANLayer(input_dim + hidden_dim, hidden_dim, sub_kan_configs, use_bias)
        self.initial_hidden = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, x):
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {D}")
        hidden = self.initial_hidden.expand(B, -1)  # [B, hidden_dim]
        # Vectorize recurrence
        hidden_expanded = hidden.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        combined = torch.cat([x, hidden_expanded], dim=-1)  # [B, T, input_dim + hidden_dim]
        outputs = self.tkan_layer(combined.view(B * T, -1)).view(B, T, self.hidden_dim)  # [B, T, hidden_dim]
        return outputs if self.return_sequences else outputs[:, -1, :]  # [B, T, hidden_dim] or [B, hidden_dim]
