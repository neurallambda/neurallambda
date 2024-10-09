# RNN/LSTM Cells that support using no nonlinearity

class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh"):
        super(MyRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.randn(hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(hidden_size))

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size).to(input.device)

        if self.bias:
            output = torch.matmul(input, self.weight_ih.t()) + self.bias_ih + torch.matmul(hidden, self.weight_hh.t()) + self.bias_hh
        else:
            output = torch.matmul(input, self.weight_ih.t()) + torch.matmul(hidden, self.weight_hh.t())

        if self.nonlinearity == "tanh":
            output = torch.tanh(output)
        elif self.nonlinearity == "relu":
            output = torch.relu(output)
        elif self.nonlinearity == 'none':
            pass
        else:
            raise ValueError("Unknown nonlinearity: {}".format(self.nonlinearity))

        return output


class FullRNN(nn.Module):
    ''' Transductive RNN, records and stacks each output, so, same sequence length as input. '''
    def __init__(self, input_size, hidden_size, bias, dropout, nonlinearity):
        super(FullRNN, self).__init__()
        self.rnn_cell = MyRNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity)
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.rnn_cell.hidden_size).to(device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            h = self.rnn_cell(x[:, t, :], h)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        # output = self.layer_norm(output)
        return output


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, forget_bias: float = 1.0):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.forget_bias = forget_bias

        # Input gate
        self.w_ii = nn.Parameter(torch.randn(hidden_size, input_size))
        self.w_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ii = nn.Parameter(torch.randn(hidden_size)) if bias else None
        self.b_hi = nn.Parameter(torch.randn(hidden_size)) if bias else None

        # Forget gate
        self.w_if = nn.Parameter(torch.randn(hidden_size, input_size))
        self.w_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_if = nn.Parameter(torch.randn(hidden_size)) if bias else None
        self.b_hf = nn.Parameter(torch.randn(hidden_size)) if bias else None

        # Cell gate
        self.w_ig = nn.Parameter(torch.randn(hidden_size, input_size))
        self.w_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ig = nn.Parameter(torch.randn(hidden_size)) if bias else None
        self.b_hg = nn.Parameter(torch.randn(hidden_size)) if bias else None

        # Output gate
        self.w_io = nn.Parameter(torch.randn(hidden_size, input_size))
        self.w_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_io = nn.Parameter(torch.randn(hidden_size)) if bias else None
        self.b_ho = nn.Parameter(torch.randn(hidden_size)) if bias else None

    def forward(self, input, state=None):
        if state is None:
            h = torch.zeros(input.size(0), self.hidden_size, device=input.device)
            c = torch.zeros(input.size(0), self.hidden_size, device=input.device)
        else:
            h, c = state

        # Input gate
        i = torch.sigmoid(F.linear(input, self.w_ii, self.b_ii) +
                          F.linear(h, self.w_hi, self.b_hi))

        # Forget gate
        f = torch.sigmoid(F.linear(input, self.w_if, self.b_if) +
                          F.linear(h, self.w_hf, self.b_hf) + self.forget_bias)

        # Cell gate
        g = torch.tanh(F.linear(input, self.w_ig, self.b_ig) +
                       F.linear(h, self.w_hg, self.b_hg))

        # Output gate
        o = torch.sigmoid(F.linear(input, self.w_io, self.b_io) +
                          F.linear(h, self.w_ho, self.b_ho))

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new
