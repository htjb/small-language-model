import pickle

import numpy as np
import torch
import yaml
from slm.networks import Transformer  # Import the Embedding class

# import onnx

np.random.seed(42)
torch.manual_seed(42)

hyperparameters = yaml.safe_load(
    open("classic_books_hyperparameters.yaml", "r")
)

vocab_model = pickle.load(open("classic_books_vocab.pkl", "rb"))

transform = Transformer(
    vocab_size=len(vocab_model.word_to_index) + 1,
    embedding_dim=hyperparameters["embedding_size"],
    mlp_layers=hyperparameters["mlp_layers"],
    mlp_dim=hyperparameters["mlp_dim"],
    context_window_size=hyperparameters["context_window_size"],
    nheads=hyperparameters["nheads"],
    predict=True,
    entropy=hyperparameters["entropy"],
)  # Create an instance of the Transformer class

state_dict = torch.load(
    "classic_books_model.pth", map_location=torch.device("cpu")
)  # Load the state dictionary

transform.load_state_dict(
    state_dict
)  # Load the state dictionary into the model

example_input = torch.randint(
    0,
    len(vocab_model.word_to_index) + 1,
    (1, hyperparameters["context_window_size"]),
).long()  # Example input tensor
print(example_input)

onnx_program = torch.onnx.export(
    transform,
    example_input,
    dynamo=True,
    f="classic_books_model.onnx",
    output_names=["output"],
)
onnx_program.save("classic_books_model.onnx")
