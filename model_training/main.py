import glob
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.model_selection import (  # Import train_test_split for splitting data
    train_test_split,
)
from slm.byte_pair_encoding import bpe  # Import the bpe class
from slm.networks import Transformer  # Import the Embedding class
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (  # Import Dataset and DataLoader for handling data
    DataLoader,
    SubsetRandomSampler,
)
from tqdm import tqdm  # Import tqdm for progress bar

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # for mac with m1 chip
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def step(
    batch: torch.Tensor,
    transform: Transformer,
    criterion: torch.nn.CrossEntropyLoss,
    entropy: bool,
):
    # batch shape: (batch_size, seq_len)
    input_seq = batch[:, :-1]  # All sequences, except last token
    target_seq = batch[:, 1:]  # Shifted targets

    out = transform(input_seq)
    output = out["output"]  # Get the output from the model
    entropy_loss = (
        out["entropy"] if entropy else 0
    )  # Get entropy loss if
    loss = (
        criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
        + entropy_loss
    )
    return loss, output, target_seq


batch_size = 16  # Define the batch size
embedding_size = 256  # Define the embedding size
mlp_layers = 1  # Define the number of MLP layers
mlp_dim = 256  # Define the MLP dimension
context_window_size = 1024  # Define the context window size
nheads = 8
entropy = True

if os.path.exists("classic_books.log"):
    os.remove("classic_books.log")

logging.basicConfig(
    filename="classic_books.log",
    level=logging.INFO,
)

hyperparameters = {
    "embedding_size": embedding_size,
    "mlp_layers": mlp_layers,
    "mlp_dim": mlp_dim,
    "context_window_size": context_window_size,
    "batch_size": batch_size,
    "nheads": nheads,
    "entropy": entropy,
}

files = glob.glob(
    "data/*.txt"
)  # Get list of all text files in the data directory
# files = ['data/alice-in-wonderland.txt']

text = []
for f in files:
    with open(f, "r") as file:
        text.append(file.readlines())  # Read the text file line by line
text = np.concatenate(text)

# Train/test/val split shuffles by default
train, test = train_test_split(text, test_size=0.3, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)

# vocab_model = bag_of_words(files)
vocab_model = bpe(train, num_merges=200)
print(f"Number of tokens: {sum(vocab_model.freqs)}")
logging.info(f"Number of tokens: {sum(vocab_model.freqs)}")
with open("classic_books_vocab.pkl", "wb") as f:
    pickle.dump(vocab_model, f)
logging.info(f"Vocabulary size: {len(vocab_model.word_to_index)}")

transform = Transformer(
    vocab_size=len(vocab_model.word_to_index) + 1,
    embedding_dim=embedding_size,
    mlp_layers=mlp_layers,
    mlp_dim=mlp_dim,
    context_window_size=context_window_size,
    nheads=nheads,
    entropy=entropy,
).to(device)  # Create an instance of the Transformer class

number_of_parameters = sum([p.numel() for p in transform.parameters()])
print("Number of paraemters: " + str(number_of_parameters))
logging.info(f"Number of Model Parameters: {number_of_parameters}")

# Assume vocab_model.codify(t) returns a 1D LongTensor for each text
train = [vocab_model.codify(t) for t in train if t.strip()]
val = [vocab_model.codify(t) for t in val if t.strip()]
test = [vocab_model.codify(t) for t in test if t.strip()]


# Collate function: pad within a batch
def collate_batch(batch):
    # batch is a list of 1D tensors
    return pad_sequence(batch, batch_first=True, padding_value=0)


# Create all indices for complete batches
num_complete_batches = len(train) // batch_size
total_indices = num_complete_batches * batch_size

# Create batch-wise shuffled indices
indices = np.arange(total_indices).reshape(-1, batch_size)
np.random.shuffle(indices)  # Shuffle rows (batches)
shuffled_indices = indices.flatten()

sampler = SubsetRandomSampler(shuffled_indices.tolist())

# DataLoaders with padding per batch
train_dataloader = DataLoader(
    train,
    batch_size=batch_size,
    shuffle=False,
    sampler=sampler,
    collate_fn=collate_batch,
)
val_dataloader = DataLoader(
    val, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
)

weights = 1.0 / (vocab_model.freqs + 1e-6)  # Inverse frequency weighting
weights = weights / weights.sum() * (len(vocab_model.word_to_index) + 1)

# prepend a zero for the PAD class
pad_weight = torch.tensor([0.0])
weights = torch.cat([pad_weight, weights])

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
optimizer = optim.AdamW(
    transform.parameters(), lr=1e-4, weight_decay=1e-5
)  # Use AdamW optimizer

best_loss = float("inf")  # Initialize best loss
best_model = None  # Placeholder for the best model
patience_counter = 0  # Initialize patience counter
patience = 50

pbar = tqdm(range(10), desc="Training Progress")  # Initialize progress bar

for epoch in pbar:  # Number of epochs
    optimizer.zero_grad()
    total_loss = 0.0
    for vector in train_dataloader:
        optimizer.zero_grad()
        vector = vector.to(device)
        loss, _, _ = step(
            vector, transform, criterion, entropy
        )  # Perform a training step
        # loss already deals with padding
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)

    val_loss = 0.0
    with torch.no_grad():
        for vector in val_dataloader:
            vector = vector.to(device)
            loss, _, _ = step(
                vector, transform, criterion, entropy
            )  # Perform a validation step
            val_loss += loss.item()
    val_loss /= len(val_dataloader)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = transform.state_dict()  # Save the best model
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(
            f"Early stopping at epoch {epoch + 1}, "
            + f"best validation loss: {best_loss}"
        )
        break

    pbar.set_postfix(
        {
            "tl": avg_loss,
            "vl": val_loss,
            "bl": best_loss,
            "pc": patience_counter,
        }
    )  # Update progress bar with current losses

if best_model is not None:
    transform.load_state_dict(best_model)  # Load the best model

torch.save(transform.state_dict(), "classic_books_model.pth")
with open("classic_books_hyperparameters.yaml", "w") as f:
    yaml.dump(hyperparameters, f)  # Save hyperparameters to a YAML file

transform.eval()
total_loss = 0.0
total_tokens = 0
correct, total_correctable = 0, 0

with torch.no_grad():
    truth, predictions = [], []
    for vector in test_dataloader:
        vector = vector.to(device)
        loss, output, target = step(vector, transform, criterion, entropy)
        output[:, :, 0] = float("-inf")  # prevent pad prediction
        pred = torch.argmax(output, dim=2)

        mask = target != 0
        correct += (pred == target).masked_select(mask).sum().item()
        total_correctable += mask.sum().item()

        # sum loss weighted by number of non-pad tokens
        total_loss += (loss * mask.sum()).item()
        total_tokens += mask.sum().item()
        truth.extend(target.masked_select(mask).cpu().tolist())
        predictions.extend(pred.masked_select(mask).cpu().tolist())

per_token_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(per_token_loss))
logging.info(f"Test Perplexity: {perplexity}")
logging.info(f"Token Accuracy: {correct / total_correctable * 100:.2f}%")

plt.scatter(truth, predictions, alpha=0.1)
plt.xlabel("True Index")
plt.ylabel("Predicted Index")
plt.title("True vs Predicted Word Indices")
plt.savefig("true_vs_predicted_indices.png")
plt.show()

out = transform(
    vocab_model.codify("Alice was beginning").unsqueeze(0).to(device)
)
output = out["output"]  # Get the output from the model
print("Output shape:", output.shape)  # Print the shape of the output
# the last ouput is the prediction for the next word
output = output.detach().cpu().numpy()
output = np.argmax(output[0, -1, 1:])
predicted_word = vocab_model.index_to_word[int(output + 1)]
print("Predicted words:", predicted_word)  # Print the predicted words
