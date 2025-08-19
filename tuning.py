import os

import numpy as np
import optuna
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
        - entropy_loss
    )
    return loss, output, target_seq


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    embedding_size = trial.suggest_categorical(
        "embedding_size", [64, 128, 256, 512]
    )  # Suggest embedding size
    mlp_layers = trial.suggest_int(
        "mlp_layers", 1, 10
    )  # Suggest number of MLP layers
    mlp_dim = trial.suggest_categorical(
        "mlp_dim", [128, 256, 512, 1024]
    )  # Suggest MLP dimension
    nheads = trial.suggest_categorical(
        "nheads", [1, 2, 4, 8, 16]
    )  # Suggest number of attention heads
    entropy = trial.suggest_categorical(
        "entropy", [True, False]
    )  # Suggest whether to compute entropy loss

    hyperparameters = {
        "embedding_size": embedding_size,
        "mlp_layers": mlp_layers,
        "mlp_dim": mlp_dim,
        "context_window_size": 1024,
        "batch_size": batch_size,
        "nheads": nheads,
        "entropy": entropy,
    }

    print(f"--- Trial Number: {trial.number} ---")  # Print the trial number
    print("Hyperparameters:", hyperparameters)  # Print the hyperparameters

    bow = bag_of_words()

    transform = Transformer(
        vocab_size=len(bow.word_to_index) + 1,
        embedding_dim=hyperparameters["embedding_size"],
        mlp_layers=hyperparameters["mlp_layers"],
        mlp_dim=hyperparameters["mlp_dim"],
        context_window_size=hyperparameters["context_window_size"],
        nheads=hyperparameters["nheads"],
        entropy=hyperparameters["entropy"],
    )  # Create an instance of the Transformer class

    with open("alice-in-wonderland.txt", "r") as file:
        text = file.readlines()  # Read the text file line by line

    # Assume bow.codify(t) returns a 1D LongTensor for each text
    codified_texts = [bow.codify(t) for t in text if t.strip()]

    # Train/test/val split
    train, test = train_test_split(
        codified_texts, test_size=0.2, random_state=42
    )
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
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=0
    )  # Ignore padding index
    optimizer = optim.AdamW(
        transform.parameters(), lr=1e-4, weight_decay=1e-5
    )  # Use AdamW optimizer

    best_loss = float("inf")  # Initialize best loss
    best_model = None  # Placeholder for the best model
    patience_counter = 0  # Initialize patience counter
    patience = 50

    pbar = tqdm(
        range(250), desc="Training Progress"
    )  # Initialize progress bar

    for epoch in pbar:  # Number of epochs
        optimizer.zero_grad()
        total_loss = 0.0
        for vector in train_dataloader:
            loss, _, _ = step(
                vector, transform, criterion, entropy
            )  # Perform a training step
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(
            train_dataloader
        )  # Average loss for the epoch

        val_loss = 0.0
        with torch.no_grad():
            for vector in val_dataloader:
                loss, _, _ = step(
                    vector, transform, criterion, entropy
                )  # Perform a validation step
                val_loss += loss.item()
            val_loss = val_loss / len(
                val_dataloader
            )  # Average validation loss

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

    transform.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss = 0.0
        correct, total = 0, 0
        for vector in test_dataloader:
            loss, output, target = step(vector, transform, criterion, entropy)
            # output: [batch, seq_len, vocab_size+1]
            output[:, :, 0] = float("-inf")  # make pad impossible to predict
            pred = torch.argmax(output, dim=2)  # [batch, seq_len]

            # Mask positions where target == 0 (padding)
            mask = target != 0
            correct += (pred == target).masked_select(mask).sum().item()
            total += mask.sum().item()
            test_loss += loss.item()
        test_loss /= len(test_dataloader)  # Average test loss
        print(f"Trial Number: {trial.number}")  # Print the trial number
        print(f"Test Loss: {test_loss}")  # Print the test loss
        print(
            f"Accuracy: {correct / (total) * 100:.2f}%"
        )  # Print the accuracy
        print(f"Correct: {correct}, Incorrect: {total - correct}")
    return correct / (total) * 100


study = optuna.create_study(direction="maximize")  # Create an Optuna study
study.optimize(objective, n_trials=5)  # Optimize the objective function
tuning_dir = "tuning_results/"  # Directory to save tuning results
os.makedirs(tuning_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the best hyperparameters
best_hyperparameters = study.best_params
with open(tuning_dir + "best_hyperparameters.yaml", "w") as f:
    yaml.dump(
        best_hyperparameters, f
    )  # Save the best hyperparameters to a YAML file
print(
    "Best Hyperparameters:", best_hyperparameters
)  # Print the best hyperparameters
torch.save(
    study.best_trial.user_attrs["model_state_dict"],
    tuning_dir + "best_model.pth",
)  # Save the best model state dictionary
print("Best Model Saved as 'best_model.pth'")  # Print confirmation of model


fig = optuna.visualization.plot_optimization_history(
    study
)  # Plot optimization history
fig.write_png(
    tuning_dir + "optimization_history.png"
)  # Save the plot to a PNG file
print(
    "Optimization history saved as 'optimization_history.png'"
)  # Print confirmation of plot
fig = optuna.visualization.plot_param_importances(
    study
)  # Plot parameter importances
fig.write_png(
    tuning_dir + "param_importances.png"
)  # Save the plot to an HTML file
print(
    "Parameter importances saved as 'param_importances.png'"
)  # Print confirmation of

optuna.study.save_study(
    study=study,
    storage="sqlite:///study.db",
    study_name=tuning_dir + "alice_in_wonderland_study",
)  # Save the study to a SQLite database
