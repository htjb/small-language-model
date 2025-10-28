import math

import torch


def sinusoidal_positional_encoding(max_seq_len, embedding_dim):
    pe = torch.zeros(max_seq_len, embedding_dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

    # Create the div_term for frequency scaling
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2).float()
        * -(math.log(10000.0) / embedding_dim)
    )

    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cosine to odd indices
    # Handle the case where embedding_dim is odd
    if embedding_dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)

    return pe
