import torch
from torch import nn
import torch.nn.functional as F




def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x




class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_head, dropout_rate):
        super(DecoderLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_head = num_head
        
        self.self_attention_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate
            )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.enc_dec_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate
            )
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(self.embed_dim, self.embed_dim*4, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask):
        y = self.self_attention_norm(x)
        y, _ = self.self_attention(query=y, key=y, value=y, attn_mask=self_mask)
        y = self.self_attention_dropout(y)
        x = x + y
        
        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y, _ = self.enc_dec_attention(query=y, key=enc_output, value=enc_output, attn_mask=i_mask)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x




class Decoder(nn.Module):
    def __init__(self, embed_dim, num_head, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(embed_dim, num_head, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            decoder_output = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask)
        return self.last_norm(decoder_output)






