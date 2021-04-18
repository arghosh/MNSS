import torch
import torch.nn as nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MonotonicGruCell(nn.Module):
    def __init__(self, input_size,  hidden_size, bias=True):
        super().__init__()
        """
        For each element in the input sequence, each layer computes the following
        function:

        MonotonicGru Math
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} +  (W_{hn}(r_t*  h_{(t-1)})+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + h_{(t-1)}
        \end{array}
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        # x is B, input_size
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size).to(device)

        gi = self.i2h(x)  # B, 3H
        gh = self.h2h(hidden)  # B, 3H
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate_tmp = i_r + h_r
        inputgate_tmp = i_i + h_i
        sigmoid = nn.Sigmoid()
        resetgate = sigmoid(resetgate_tmp)
        inputgate = sigmoid(inputgate_tmp)
        hr = self.h2h(hidden * resetgate)
        _, _, h_n = hr.chunk(3, 1)
        newgate = sigmoid(i_n + h_n)
        hy = hidden + (1.-hidden) * inputgate * newgate
        return hy


class MonotonicGru(nn.Module):
    def __init__(self, input_size,  hidden_size, bias=True, num_layers=1, batch_first=False, dropout=0.0):
        super().__init__()
        self.cell = MonotonicGruCell(
            input_size=input_size, hidden_size=hidden_size, bias=True)
        self.batch_first = batch_first

    def forward(self, input_, lengths, hidden=None):
        # input_ is of dimensionalty (T, B, input_size, ...)
        # lenghths is B,
        dim = 1 if self.batch_first else 0
        outputs = []
        for x in torch.unbind(input_, dim=dim):  # x dim is B, I
            hidden = self.cell(x, hidden)
            outputs.append(hidden.clone())

        hidden_states = torch.stack(outputs)  # T, B, H
        last_states = []
        for idx, l in enumerate(lengths):
            last_states.append(hidden_states[l-1, idx, :])
        last_states = torch.stack(last_states)
        return hidden_states, last_states
