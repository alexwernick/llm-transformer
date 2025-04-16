import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Create an empty tensor of shape [max_seq_length, d_model]
        # to store positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        # Create a column vector containing position indices from 0 to
        # max_seq_length-1, with shape [max_seq_length, 1].
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Compute the division terms used in the sine/cosine calculations.
        # This creates frequencies that decrease geometrically from 1 to 1/10000.
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # noqa: E501 Register the positional encoding as a buffer (persistent state that's not a parameter).
        # Add a batch dimension, making shape [1, max_seq_length, d_model].
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # takes all positional encodings up to the size of the
        # sequence length x.size(1)
        return x + self.pe[:, : x.size(1)]
