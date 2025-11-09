import pickle

import torch
import yaml
from slm.lstm import LSTM, MLP, Embedding

model_name = "simple-wiki-lstm"

hyperparameters = yaml.safe_load(
    open(model_name + "_hyperparameters.yaml", "r")
)

vocab_model = pickle.load(open(model_name + "_vocab.pkl", "rb"))

embedder = Embedding(
    hyperparameters["embedding_size"], len(vocab_model.word_to_index) + 1
)
lstm = LSTM(hyperparameters["embedding_size"], num_layers=hyperparameters["lstm_layers"])
mlp = MLP(
    hyperparameters["embedding_size"],
    hyperparameters["mlp_layers"],
    hyperparameters["mlp_dim"],
    len(vocab_model.word_to_index) + 1,
)

embedder_state_dict = torch.load(
    model_name + "_embedder.pth", map_location=torch.device("cpu")
)  # Load the state dictionary

lstm_state_dict = torch.load(
    model_name + "_lstm.pth", map_location=torch.device("cpu")
)  # Load the state dictionary

mlp_state_dict = torch.load(
    model_name + "_mlp.pth", map_location=torch.device("cpu")
)  # Load the state dictionary

embedder.load_state_dict(embedder_state_dict)
lstm.load_state_dict(lstm_state_dict)
mlp.load_state_dict(mlp_state_dict)

embedder.eval()
lstm.eval()
mlp.eval()

test_phrase = "what is "  # Define a test phrase

vector = vocab_model.codify(test_phrase).unsqueeze(0)  # [:-1]

h = torch.zeros(hyperparameters['lstm_layers'], 1, hyperparameters["embedding_size"])
c = torch.zeros(hyperparameters['lstm_layers'], 1, hyperparameters["embedding_size"])

embedded = embedder(vector)  # Single token

out, h, c = lstm(embedded[:, :-1, :], h, c)

input_token = vector[:, -1].unsqueeze(0)

output = []
for _ in range(100):
    embedded = embedder(input_token)  # Single token
    h, c = h.reshape(hyperparameters['lstm_layers'], 1, -1), c.reshape(hyperparameters['lstm_layers'], 1, -1)
    out, h, c = lstm(embedded, h, c)
    next_token = mlp(out).argmax(dim=-1)  # could do top-k here
    input_token = next_token
    output.append(next_token.item())
    if next_token.item() == vocab_model.word_to_index["EOS"]:
        break

decoded_output = [vocab_model.index_to_word[int(o)] for o in output]
print(decoded_output)
print(test_phrase + " " + "".join(decoded_output))
