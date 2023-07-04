import torch
import numpy as np
import torch.nn as nn

import cv2

# class PositionalEncoding(nn.Module):
#     "Implement the PE function."
#     def __init__(self, d_model, dropout, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) *
#                              -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#         
#     def forward(self, x):
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         return self.dropout(x)



class PositionalEncoding(nn.Module):
 
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
 
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
 
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
 
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
 
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        
        ######## visualize the 'sinusoid_table' in here
        #table_visualize = np.expand_dims(sinusoid_table, axis=2)
        #cv2.imshow('visualize', table_visualize)
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
 
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
    def forward_with_idx(self, x, idx):
        return x + self.pos_table[:, idx].unsqueeze(dim=1).clone().detach()
    
    