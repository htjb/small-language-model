from datasets import load_dataset
from tqdm import tqdm

# This grabs the 20220301 English dump (you can choose a newer date if you like)
wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split=None)[
    "train"
]

pbar = tqdm(wiki, desc="Processing articles")
for i, example in enumerate(pbar):
    with open(f"wiki_{i}.txt", "w", encoding="utf-8") as f:
        f.write(example["text"])
    if i >= 10000:  # Limit to first 1000 articles for this example
        break
