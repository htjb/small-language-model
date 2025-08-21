import re

import numpy as np
import torch


class bag_of_words:
    def __init__(self):
        text = []  # Initialize an empty list to hold the text lines
        with open("alice-in-wonderland.txt", "r") as file:
            text = file.readlines()  # Read the text file line by line
        text = [
            line.strip().lower() for line in text if line.strip()
        ]  # Clean and filter empty lines
        self.text = [
            re.sub(r"[^\w\s]", "", line) for line in text
        ]  # Remove punctuation

        words = np.unique(
            np.concatenate([line.split() for line in self.text])
        )  # Get unique words
        words = np.append(words, "UNK")
        self.word_to_index = {
            word: i + 1 for i, word in enumerate(words)
        }  # Create a mapping from words to indices

    def codify(self, line):
        """
        Convert a line of text into a bag-of-words representation.
        """
        line = line.strip().lower()  # Clean the input line
        words = line.split()
        words = [re.sub(r"[^\w\s]", "", word) for word in words]
        # look up the index of each word in the mapping
        # unkonwn words will be mapped to UNK
        indices = torch.tensor(
            [
                self.word_to_index.get(word, self.word_to_index["UNK"])
                for word in words
            ]
        )
        return indices
