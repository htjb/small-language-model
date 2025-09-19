import glob
import logging
import os
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import yaml
from sklearn.model_selection import (  # Import train_test_split for splitting data
    train_test_split,
)
from slm.byte_pair_encoding import bpe  # Import the bpe class
from slm.networks import StackedTransformers
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from torch.utils.data import (  # Import Dataset and DataLoader for handling data
    DataLoader,
    SubsetRandomSampler,
)
from tqdm import tqdm  # Import tqdm for progress bar
import unicodedata


def clean_non_latin(text):
    # 1. Decompose accents so é → e + ́
    nfkd = unicodedata.normalize("NFKD", text)
    # 2. Remove combining marks (accents)
    no_accents = "".join(c for c in nfkd
                         if not unicodedata.combining(c))
    # 3. Keep only characters you want: 
    #    - ASCII letters/numbers
    #    - basic punctuation/math (common Unicode math symbols)
    allowed = re.sub(r"[^A-Za-z0-9\s\.,;:\-\+\*/=<>\(\)\[\]\{\}~!@#\$%\^&\|\\\?\^\_]", "", no_accents)
    return allowed

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # for mac with m1 chip
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0  # constant after warmup (or replace with decay fn)

    return LambdaLR(optimizer, lr_lambda)


def step(
    batch: torch.Tensor,
    transform: StackedTransformers,
    criterion: torch.nn.CrossEntropyLoss,
    entropy: bool,
):
    # batch shape: (batch_size, seq_len)
    input_seq = batch[:, :-1]  # All sequences, except last token
    target_seq = batch[:, 1:]  # Shifted targets

    out = transform(input_seq)
    output = out["output"]  # Get the output from the model
    entropy_loss = out["entropy"] if entropy else 0  # Get entropy loss if
    loss = (
        criterion(output.reshape(-1, output.size(-1)), target_seq.reshape(-1))
        + entropy_loss
    )
    return loss, output, target_seq


batch_size = 32  # Define the batch size
embedding_size = 128  # Define the embedding size
mlp_layers = 1  # Define the number of MLP layers
mlp_dim = 128  # Define the MLP dimension
context_window_size = 1024  # Define the context window size
nheads = 2
ntransformers = 1
entropy = False
model_name = "simple-wiki"
load_vocab = False

if os.path.exists(model_name + ".log"):
    os.remove(model_name + ".log")

logging.basicConfig(
    filename=model_name + ".log",
    level=logging.INFO,
)

hyperparameters = {
    "embedding_size": embedding_size,
    "mlp_layers": mlp_layers,
    "mlp_dim": mlp_dim,
    "context_window_size": context_window_size,
    "batch_size": batch_size,
    "nheads": nheads,
    "ntransformers": ntransformers,
    "entropy": entropy,
}

files = glob.glob(
    "data/" + model_name + "/*.txt"
)  # Get list of all text files in the data directory
# files = ['data/alice-in-wonderland.txt']

text = []
for f in files:
    with open(f, "r") as file:
        text.append(file.readlines())  # Read the text file line by line
text = np.concatenate(text)
text = [re.split(r'(?<=[.!?])\s+', line.strip()) 
        for line in text if line.strip()]  # Remove empty lines and split on punctuation
text = np.concatenate(text).tolist()
text = [clean_non_latin(t) for t in text]

while any(len(t) > context_window_size for t in text):
    new_text = []
    for i, t in enumerate(text):
        if len(t) > context_window_size:
            # split at nearest space before context_window_size
            split_idx = t.rfind(' ', 0, context_window_size)
            if split_idx == -1:  # no space found, hard split
                split_idx = context_window_size
            new_text.append(t[:split_idx].strip())
            new_text.append(t[split_idx:].strip())
        else:
            new_text.append(t)
    text = new_text
            
# Train/test/val split shuffles by default
train, test = train_test_split(text, test_size=0.3, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)
print(f"Train size: {len(train)}")

# vocab_model = bag_of_words(files)
if load_vocab and os.path.exists(model_name + "_vocab.pkl"):
    with open(model_name + "_vocab.pkl", "rb") as f:
        vocab_model = pickle.load(f)
    print(f"Loaded vocabulary of size: {len(vocab_model.word_to_index)}")
    logging.info(f"Loaded vocabulary of size: {len(vocab_model.word_to_index)}")
else:
    vocab_model = bpe(train[:int(len(train)/100*5)], num_merges=1000)
print(f"Number of tokens: {sum(vocab_model.freqs)}")
logging.info(f"Number of tokens: {sum(vocab_model.freqs)}")
with open(model_name + "_vocab.pkl", "wb") as f:
    pickle.dump(vocab_model, f)
logging.info(f"Vocabulary size: {len(vocab_model.word_to_index)}")

merger_rules = vocab_model.merger_rules
keys = list(merger_rules.keys())
np.savetxt(
    "../website/assets/" + model_name + "_merger_rules.txt", keys, fmt="%s"
)

with open("../website/assets/" + model_name + "_word_to_index.yaml", "w") as f:
    yaml.dump(vocab_model.word_to_index, f)

with open("../website/assets/" + model_name + "_index_to_word.yaml", "w") as f:
    yaml.dump(vocab_model.index_to_word, f)


transform = StackedTransformers(
    vocab_size=len(vocab_model.word_to_index) + 1,
    embedding_dim=embedding_size,
    mlp_layers=mlp_layers,
    mlp_dim=mlp_dim,
    context_window_size=context_window_size,
    nheads=nheads,
    entropy=entropy,
    ntransformers=ntransformers,
).to(device)  # Create an instance of the Transformer class

"""from torchsummary import summary
print(summary(transform, (batch_size, context_window_size - 1)))
exit()"""

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
optimizer = optim.AdamW(transform.parameters(), lr=1e-3, weight_decay=0.01)
scaler = GradScaler(device.type)

best_loss = float("inf")  # Initialize best loss
best_model = None  # Placeholder for the best model
patience_counter = 0  # Initialize patience counter
patience = 50
epochs = 250

total_steps = epochs * len(train_dataloader) / batch_size
warmup_steps = 2 * len(train_dataloader) / batch_size
scheduler = get_scheduler(
    optimizer, warmup_steps=warmup_steps, total_steps=total_steps
)

pbar = tqdm(range(epochs), desc="Training Progress")  # Initialize progress bar

for epoch in pbar:  # Number of epochs
    total_loss = 0.0
    for vector in train_dataloader:
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            vector = vector.to(device)
            loss, _, _ = step(
                vector, transform, criterion, entropy
            )  # Perform a training step
            # loss already deals with padding
            total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)#.step()
        scaler.update()
        scheduler.step()
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

torch.save(transform.state_dict(), model_name + "_model.pth")
with open(model_name + "_hyperparameters.yaml", "w") as f:
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
