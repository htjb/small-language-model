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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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
        embedding = self.layer_norm(embedding)

        # Create padding mask: True where pad tokens are
        pad_mask = (x == 0)  # shape [batch, seq_len]

        attention_head_outputs = []
        attention_head_weights = []
        for i in range(self.nheads):
            query = self.query[i](embedding)
            key = self.key[i](embedding)
            value = self.value[i](embedding)

            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
                key.size(-1) ** 0.5
            )

            # Causal mask
            seq_len = x.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
            
            # Combine causal + pad mask
            # pad_mask: [batch, seq_len] -> expand to [batch, 1, seq_len]
            combined_mask = (causal_mask).unsqueeze(0) | pad_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(combined_mask, -1e9)

            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_head_weights.append(attention_weights)

            attention_head_outputs.append(torch.matmul(attention_weights, value))

        attention_head_weights = torch.stack(attention_head_weights, dim=1)  # shape: [batch_size, nheads, seq_len, seq_len]
        # row-wise entropy
        entropy = -torch.sum(attention_head_weights * torch.log(attention_head_weights + 1e-8), axis=-1)
        # shape: [batch_size, nheads, seq_len]
        # average to a scalar
        entropy_loss = torch.mean(entropy)
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

        if self.predict:
            return x
        else:
            return x, entropy_loss    
