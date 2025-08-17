import torch.nn as nn
import torch
import math


def sinusoidal_positional_encoding(max_seq_len, embedding_dim):
    pe = torch.zeros(max_seq_len, embedding_dim)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    
    # Create the div_term for frequency scaling
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                         -(math.log(10000.0) / embedding_dim))
    
    # Apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    
    # Apply cosine to odd indices
    # Handle the case where embedding_dim is odd
    if embedding_dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        mlp_layers,
        mlp_dim,
        context_window_size,
        nheads,
        predict=False
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.nheads = nheads
        self.predict = predict


        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        for _ in range(nheads):
            self.query.append(nn.Linear(embedding_dim, embedding_dim))
            self.key.append(nn.Linear(embedding_dim, embedding_dim))
            self.value.append(nn.Linear(embedding_dim, embedding_dim))

        self.pos_enc = sinusoidal_positional_encoding(
            context_window_size, embedding_dim
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim, mlp_dim))
        for _ in range(mlp_layers):
            self.layers.append(nn.Linear(mlp_dim, mlp_dim))
        self.layers.append(nn.Linear(mlp_dim, vocab_size))

    def forward(self, x):
        embedding = self.embedding(x) + self.pos_enc[: x.size(1)]
        # layer normalization
        embedding = self.layer_norm(embedding)

        attention_head_outputs = []
        for i in range(self.nheads):
            query = self.query[i](embedding)
            key = self.key[i](embedding)
            value = self.value[i](embedding)
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
                key.size(-1) ** 0.5
            )
            # mask self-attention
            # Create a mask to allow each token to attend only to itself and future tokens (causal mask)
            mask = torch.tril(torch.ones(attention_scores.size(-2), 
                                        attention_scores.size(-1)), diagonal=-1).bool()
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))
            attention_weights = torch.nn.functional.softmax(
                attention_scores, dim=-1
            )
            attention_head_outputs.append(
                torch.matmul(attention_weights, value)
            )

        x = torch.stack(attention_head_outputs, dim=1).sum(dim=1)
        x = x + embedding
        x = self.layer_norm(x)  # Apply layer normalization
        
        if self.predict:
            x = x[:, -1, :]
            x = x.unsqueeze(1)
        
        x = torch.nn.functional.relu(x)  # Apply activation function

        # First layer (no residual for input)
        x = self.layers[0](x)

        # Residual connections for intermediate layers
        for i in range(1, len(self.layers) - 1):
            residual = x
            x = self.layers[i](x)
            x = torch.nn.functional.relu(x)
            x = x + residual  # Add residual connection

        # Final layer (output projection, no residual)
        x = self.layers[-1](x)

        return x
