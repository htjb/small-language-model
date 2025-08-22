import re

import numpy as np
import torch


class bag_of_words:
    def __init__(self):
        text = []  # Initialize an empty list to hold the text lines
        with open("alice-in-wonderland.txt", "r") as file:
            text = file.readlines()  # Read the text file line by line

        # tokenize into words + punctuation
        # | is or operator in regex
        # \w+ matches sequences of word characters (letters, digits, underscores)
        # [] represents a set, ^ negates the set, \w matches word characters, \s matches whitespace characters
        # so [^\w\s] matches any character that is not a word character or whitespace
        tokenized = [re.findall(r"\w+|[^\w\s]", line) for line in text]

        # flatten and get unique tokens
        words = np.unique(np.concatenate(tokenized))
        words = np.append(words, "UNK")

        self.word_to_index = {word: i + 1 for i, word in enumerate(words)}

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
