import math

import torch
import torch.nn as nn


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


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        mlp_layers,
        mlp_dim,
        context_window_size,
        nheads,
        predict=False,
        entropy=False,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.nheads = nheads
        self.predict = predict
        self.entropy = entropy  # Whether to compute entropy loss

        if self.entropy:
            self.entropy_weight = nn.Parameter(
                torch.tensor(0.1), requires_grad=True
            )

        self.query = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()
        for _ in range(nheads):
            self.query.append(
                nn.Linear(embedding_dim // nheads, embedding_dim // nheads)
            )
            self.key.append(
                nn.Linear(embedding_dim // nheads, embedding_dim // nheads)
            )
            self.value.append(
                nn.Linear(embedding_dim // nheads, embedding_dim // nheads)
            )

        self.pos_enc = sinusoidal_positional_encoding(
            context_window_size, embedding_dim
        )

        self.layer_norm_pre_attention = nn.LayerNorm(embedding_dim)
        self.layer_norm_post_attention = nn.LayerNorm(embedding_dim)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(embedding_dim, mlp_dim))
        for _ in range(mlp_layers):
            self.layers.append(nn.Linear(mlp_dim, mlp_dim))
        self.layers.append(nn.Linear(mlp_dim, vocab_size))

    def forward(self, x):
        embedding = self.embedding(x) + self.pos_enc[: x.size(1)].to(x.device)
        layer_normed_embedding = self.layer_norm_pre_attention(embedding)
        normed_embedding = []
        for i in range(self.nheads):
            normed_embedding.append(
                layer_normed_embedding[
                    :,
                    :,
                    (i * layer_normed_embedding.size(2)) // self.nheads : (
                        (i + 1) * layer_normed_embedding.size(2)
                    )
                    // self.nheads,
                ]
            )

        # Create padding mask: True where pad tokens are
        pad_mask = x == 0  # shape [batch, seq_len]

        attention_head_outputs = []
        attention_head_weights = []
        for i in range(self.nheads):
            query = self.query[i](normed_embedding[i])
            key = self.key[i](normed_embedding[i])
            value = self.value[i](normed_embedding[i])

            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
                key.size(-1) ** 0.5
            )

            # Causal mask
            seq_len = normed_embedding[i].size(1)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).bool()

            # Combine causal + pad mask
            # pad_mask: [batch, seq_len] -> expand to [batch, 1, seq_len]
            combined_mask = (causal_mask).unsqueeze(0) | pad_mask.unsqueeze(1)

            attention_scores = attention_scores.masked_fill(
                combined_mask, -1e9
            )

            attention_weights = torch.nn.functional.softmax(
                attention_scores, dim=-1
            )
            attention_head_weights.append(attention_weights)

            attention_head_outputs.append(
                torch.matmul(attention_weights, value)
            )

        attention_head_weights = torch.stack(
            attention_head_weights, dim=1
        )  # shape: [batch_size, nheads, seq_len, seq_len]
        # row-wise entropy
        if self.entropy and self.predict is False:
            uniform = torch.full_like(attention_head_weights, 1.0 / 
                        attention_head_weights.size(-1))
            kl = torch.sum(
                attention_head_weights * 
                (torch.log(attention_head_weights + 1e-8) - 
                math.log(1.0 / attention_head_weights.size(-1))),
                dim=-1
            )
            entropy_loss = kl.mean()   # >= 0 and well-scaled

        # x = torch.stack(attention_head_outputs, dim=1).sum(dim=1)
        x = torch.cat(attention_head_outputs, dim=-1)

        x = x + embedding
        x = self.layer_norm_post_attention(x)  # Apply layer normalization

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

        if self.entropy and self.predict is False:
            return {
                "output": x,
                "entropy": entropy_loss * torch.abs(self.entropy_weight),
            }
        else:
            return {"output": x, "entropy": None}

class StackedTansformers(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        mlp_layers,
        mlp_dim,
        context_window_size,
        nheads,
        ntransformers,
        predict=False,
        entropy=False,
    ):
        super(StackedTansformers, self).__init__()
        self.transformers = nn.ModuleList()
        for _ in range(ntransformers):
            self.transformers.append(
                Transformer(
                    vocab_size,
                    embedding_dim,
                    mlp_layers,
                    mlp_dim,
                    context_window_size,
                    nheads,
                    predict=predict,
                    entropy=entropy,
                )
            )

    def forward(self, x):
        entropy_loss = 0
        for transformer in self.transformers:
            out = transformer(x)
            x = out["output"]
            if transformer.entropy and transformer.predict is False:
                entropy_loss += out["entropy"]
        return {"output": x, "entropy": entropy_loss}