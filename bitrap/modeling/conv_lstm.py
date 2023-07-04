from torch import nn
import torch

#### implementation 1
class ConvLSTM(nn.Module):
    def __init__(self, input_channel, num_filter, b_h_w, kernel_size, stride=1, padding=1, device='cuda'):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=input_channel + num_filter,
                               out_channels=num_filter*4,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.device = device
        self._batch_size, self._state_height, self._state_width = b_h_w
        # if using requires_grad flag, torch.save will not save parameters in deed although it may be updated every epoch.
        # Howerver, if you use declare an optimizer like Adam(model.parameters()),
        # parameters will not be updated forever.
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, self._state_height, self._state_width)).to(device)
        self._input_channel = input_channel
        self._num_filter = num_filter

    # inputs and states should not be all none
    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=60):

        if states is None:
            c = torch.zeros((inputs.size(1), self._num_filter, self._state_height, self._state_width), dtype=torch.float).to(self.device)
            h = torch.zeros((inputs.size(1), self._num_filter, self._state_height, self._state_width), dtype=torch.float).to(self.device)
        else:
            h, c = states
        
        outputs = []
        for index in range(seq_len):
            # initial inputs
            if inputs is None:
                x = torch.zeros((h.size(0), self._input_channel, self._state_height, self._state_width), dtype=torch.float).to(self.device)
            else:
                x = inputs[index, ...]
            cat_x = torch.cat([x, h], dim=1)
            conv_x = self._conv(cat_x)
            
            i, f, tmp_c, o = torch.chunk(conv_x, 4, dim=1)
            
            i = torch.sigmoid(i+self.Wci*c)
            f = torch.sigmoid(f+self.Wcf*c)
            c = f*c + i*torch.tanh(tmp_c)
            o = torch.sigmoid(o+self.Wco*c)
            h = o*torch.tanh(c)
            outputs.append(h)
        
        return torch.stack(outputs), (h, c)







#### implementation 2
class CLSTM(nn.Module):
    
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.input_channels + self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding),
#             #nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features)
#             )    # the ability to extract features may not be strong. Thus, make it deeper.
        
        #
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, 2 * self.num_features, self.filter_size, 1, self.padding),
            nn.BatchNorm2d(2 * self.num_features),
            nn.Conv2d(2 * self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding),
            nn.BatchNorm2d(4 * self.num_features)
            )    # the ability to extract features may not be strong. Thus, make it deeper.
        

    def forward(self, inputs=None, hidden_state=None):
        #
        if hidden_state is None:
            hx = torch.zeros(inputs.shape[0], self.num_features, self.shape[0], self.shape[1]).cuda()
            cx = torch.zeros(inputs.shape[0], self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        seq_len = inputs.shape[1]
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.shape[0], self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[:, index, :]
            
            combined = torch.cat((x, hx), dim=1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)






class CGRU(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, 2 * self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels, self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev), 1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext





