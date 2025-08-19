import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.model_selection import (  # Import train_test_split for splitting data
    train_test_split,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (  # Import Dataset and DataLoader for handling data
    DataLoader,
    SubsetRandomSampler,
)
from tqdm import tqdm  # Import tqdm for progress bar

from slm.bag_of_words import bag_of_words  # Import the bag_of_words class
from slm.networks import Transformer  # Import the Embedding class


def step(
    batch: torch.Tensor,
    transform: Transformer,
    criterion: torch.nn.CrossEntropyLoss,
):
    # batch shape: (batch_size, seq_len)
    input_seq = batch[:, :-1]  # All sequences, except last token
    target_seq = batch[:, 1:]  # Shifted targets

    output, entropy_loss = transform(
        input_seq
    )  # shape: (batch_size, seq_len-1, vocab_size)

    # CrossEntropyLoss expects (N, C) and (N,) => flatten batch and seq dims
    loss = (
        criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
        - entropy_loss
    )
    return loss, output, target_seq


batch_size = 64  # Define the batch size
embedding_size = 128  # Define the embedding size
mlp_layers = 2  # Define the number of MLP layers
mlp_dim = 512  # Define the MLP dimension
context_window_size = 512  # Define the context window size
nheads = 10

hyperparameters = {
    "embedding_size": embedding_size,
    "mlp_layers": mlp_layers,
    "mlp_dim": mlp_dim,
    "context_window_size": context_window_size,
    "batch_size": batch_size,
    "nheads": nheads,  # Define the number of attention heads
}

bow = bag_of_words()

transform = Transformer(
    vocab_size=len(bow.word_to_index) + 1,
    embedding_dim=embedding_size,
    mlp_layers=mlp_layers,
    mlp_dim=mlp_dim,
    context_window_size=context_window_size,
    nheads=nheads,
)  # Create an instance of the Transformer class

with open("alice-in-wonderland.txt", "r") as file:
    text = file.readlines()  # Read the text file line by line

# Assume bow.codify(t) returns a 1D LongTensor for each text
codified_texts = [bow.codify(t) for t in text if t.strip()]

# Train/test/val split
train, test = train_test_split(codified_texts, test_size=0.3, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)


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

sampler = SubsetRandomSampler(shuffled_indices)

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

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
optimizer = optim.AdamW(
    transform.parameters(), lr=1e-4, weight_decay=1e-5
)  # Use AdamW optimizer

best_loss = float("inf")  # Initialize best loss
best_model = None  # Placeholder for the best model
patience_counter = 0  # Initialize patience counter
patience = 50

pbar = tqdm(range(1000), desc="Training Progress")  # Initialize progress bar

for epoch in pbar:  # Number of epochs
    optimizer.zero_grad()
    total_loss = 0.0
    for vector in train_dataloader:
        loss, _, _ = step(
            vector, transform, criterion
        )  # Perform a training step
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)  # Average loss for the epoch

    val_loss = 0.0
    with torch.no_grad():
        for vector in val_dataloader:
            loss, _, _ = step(
                vector, transform, criterion
            )  # Perform a validation step
            val_loss += loss.item()
        val_loss = val_loss / len(val_dataloader)  # Average validation loss

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
            "avg_loss": avg_loss,
            "val_loss": val_loss,
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        }
    )  # Update progress bar with current losses

transform.load_state_dict(best_model)  # Load the best model

torch.save(transform.state_dict(), "alice_in_wonderland_model.pth")
with open("alice_in_wonderland_hyperparameters.yaml", "w") as f:
    yaml.dump(hyperparameters, f)  # Save hyperparameters to a YAML file

transform.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_loss = 0.0
    correct, total = 0, 0
    for vector in test_dataloader:
        loss, output, target = step(vector, transform, criterion)
        # output: [batch, seq_len, vocab_size+1]
        output[:, :, 0] = float("-inf")  # make pad impossible to predict
        pred = torch.argmax(output, dim=2)  # [batch, seq_len]

        # Mask positions where target == 0 (padding)
        mask = target != 0
        correct += (pred == target).masked_select(mask).sum().item()
        total += mask.sum().item()
        test_loss += loss.item()
    test_loss /= len(test_dataloader)  # Average test loss
    print(f"Test Loss: {test_loss}")  # Print the test loss
    print(f"Accuracy: {correct / (total) * 100:.2f}%")  # Print the accuracy
    print(
        f"Correct: {correct}, Incorrect: {total - correct}"
    )  # Print the number of correct and incorrect predictions
print("Vocabulary size:", len(bow.word_to_index))  # Print the vocabulary size

# only need to make pass through the mlp for the last word... will need to think
# about how to do this in the future
output, _ = transform(
    torch.tensor(bow.codify("Alice was beginning")).unsqueeze(0)
)
print("Output shape:", output.shape)  # Print the shape of the output
# the last ouput is the prediction for the next word
output = np.argmax(output[0, -1, 1:].detach().numpy())
index_to_word = {i: w for w, i in bow.word_to_index.items()}
predicted_word = index_to_word[output]
print("Predicted words:", predicted_word)  # Print the predicted words
