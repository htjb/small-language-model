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
        out_size,
        embedding_dim,
        mlp_layers,
        mlp_dim,
        context_window_size,
        nheads,
        predict=False,
        entropy=False,
        embedding=True,
    ):
        super(Transformer, self).__init__()
        if embedding:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
        else:
            self.embedding = False
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
        self.layers.append(nn.Linear(mlp_dim, out_size))

    def forward(self, x):
        if self.embedding:
            embedding = self.embedding(x) + self.pos_enc[: x.size(1)].to(
                x.device
            )
            layer_normed_embedding = self.layer_norm_pre_attention(embedding)
        else:
            embedding = x  # for residual path
            layer_normed_embedding = x
        normed_embedding = []
        for i in range(self.nheads):
            normed_embedding.append(
                layer_normed_embedding[
                    :,  # batch size
                    :,  # sequence length
                    (i * layer_normed_embedding.size(2)) // self.nheads : (
                        (i + 1) * layer_normed_embedding.size(2)
                    )
                    // self.nheads,  # embedding dim//nheads
                ]
            )

        if self.embedding:
            pad_mask = x == 0  # Assuming padding token index is 0

        attention_head_outputs = []
        entropy_loss = 0
        for i in range(self.nheads):
            query = self.query[i](normed_embedding[i])
            key = self.key[i](normed_embedding[i])
            value = self.value[i](normed_embedding[i])

            # rows = query, cols = key??
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
                key.size(-1) ** 0.5
            )

            # Causal mask
            # Create a upper triangular matrix for causal masking
            seq_len = normed_embedding[i].size(1)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()

            # Combine causal + pad mask
            # pad_mask: [batch, seq_len] -> expand to [batch, 1, seq_len] -> [batch, seq_len, seq_len]
            key_padding_mask = (
                pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
                if self.embedding
                else None
            )
            if self.embedding:
                combined_mask = (~causal_mask).unsqueeze(0) | (
                    key_padding_mask
                )
            else:
                combined_mask = ~causal_mask.unsqueeze(0)

            # masks the true positions with a large negative value
            attention_scores = attention_scores.masked_fill(
                combined_mask, -1e9
            )

            attention_weights = torch.nn.functional.softmax(
                attention_scores, dim=-1
            )
            if self.entropy:
                kl_list = []
                for w in attention_weights:  # w is [914, 914]
                    kl_list = w * (torch.log(w + 1e-8) - math.log(1.0 / w.size(-1)))
                kl = torch.stack(kl_list)
                entropy_loss += kl.mean()  # >= 0 and well-scaled

            attention_head_outputs.append(
                torch.matmul(attention_weights, value)
            )

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


class StackedTransformers(nn.Module):
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
        super(StackedTransformers, self).__init__()
        self.transformers = nn.ModuleList()
        for i in range(ntransformers):
            if i == 0:
                embed_flag = True
                in_size = vocab_size  # used only for embedding layer
            else:
                embed_flag = False
                in_size = embedding_dim  # float embeddings

            is_last = i == ntransformers - 1
            out_size = vocab_size if is_last else embedding_dim

            self.transformers.append(
                Transformer(
                    vocab_size=in_size,
                    out_size=out_size,
                    embedding_dim=embedding_dim,
                    mlp_layers=mlp_layers,
                    mlp_dim=mlp_dim,
                    context_window_size=context_window_size,
                    nheads=nheads,
                    predict=predict,
                    entropy=entropy,
                    embedding=embed_flag,
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
