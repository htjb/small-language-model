import math
import re
import unicodedata

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


def clean_non_latin(text):
    # 1. Decompose accents so é → e + ́
    nfkd = unicodedata.normalize("NFKD", text)
    # 2. Remove combining marks (accents)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # 3. Keep only characters you want:
    #    - ASCII letters/numbers
    #    - basic punctuation/math (common Unicode math symbols)
    allowed = re.sub(
        r"[^A-Za-z0-9\s\.,;:\-\+\*/=<>\(\)\[\]\{\}~!@#\$%\^&\|\\\?\^\_]",
        "",
        no_accents,
    )
    return allowed


def split_at_context_window(text, context_window_size, space_token_id):
    """
    text: list of 1D tensors of token IDs
    context_window_size: max tokens per chunk
    space_token_id: token ID of space for soft splitting
    """
    result = []

    for t in text:  # iterate over each sequence
        start = 0
        n = t.size(0)
        while start < n:
            end = min(start + context_window_size, n)
            chunk = t[start:end]

            # try to split at last space in the chunk if possible
            if end < n:
                space_positions = (chunk == space_token_id).nonzero(
                    as_tuple=True
                )[0]
                if len(space_positions) > 0:
                    split_idx = space_positions[-1].item() + 1
                    chunk = chunk[:split_idx]
                    end = start + split_idx

            result.append(chunk)
            start = end

    return result  # list of 1D tensors
