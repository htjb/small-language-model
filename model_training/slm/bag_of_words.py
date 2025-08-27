import re
from collections import Counter

import numpy as np
import torch


class bag_of_words:
    def __init__(self, files):
        text = []  # Initialize an empty list to hold the text lines
        for f in files:
            with open(f, "r") as file:
                text = file.readlines()  # Read the text file line by line

        # tokenize into words + punctuation
        # | is or operator in regex
        # \w+ matches sequences of word characters (letters, digits, underscores)
        # [] represents a set, ^ negates the set, \w matches word characters, \s matches whitespace characters
        # so [^\w\s] matches any character that is not a word character or whitespace
        tokenized = [
            re.findall(r"\w+|[^\w\s]", line, flags=re.UNICODE) for line in text
        ]

        words = np.concatenate(tokenized)
        words = [w.strip("_") for w in words]
        # add EOS at sentence ends (same logic as codify)
        processed_words = []
        for w in words:
            processed_words.append(w)
            if w in [".", "!", "?"]:
                processed_words.append("EOS")
        processed_words.append("EOS")

        # count frequencies
        counter = Counter(processed_words)

        # build vocab
        vocab = sorted(counter.keys())
        vocab.append("UNK")  # add UNK explicitly if not already there

        self.word_to_index = {word: i + 1 for i, word in enumerate(vocab)}
        self.index_to_word = {i + 1: word for i, word in enumerate(vocab)}

        # store frequencies as a tensor aligned with vocab indices
        self.freqs = torch.tensor(
            [counter.get(word, 0) for word in vocab], dtype=torch.float
        )

    def codify(self, line):
        """
        Convert a line of text into a bag-of-words representation.
        """
        # remove leading/trailing whitespace
        line = line.strip()
        words = re.findall(r"\w+|[^\w\s]", line)
        # look up the index of each word in the mapping
        # unkonwn words will be mapped to UNK

        # insert EOS after ., !, ?
        new_tokens = []
        for w in words:
            new_tokens.append(w)
            if w in [".", "!", "?"]:
                new_tokens.append("EOS")

        # ensure the sequence always ends with EOS
        if not new_tokens or new_tokens[-1] != "EOS":
            new_tokens.append("EOS")

        indices = torch.tensor(
            [
                self.word_to_index.get(tok, self.word_to_index["UNK"])
                for tok in new_tokens
            ]
        )

        return indices
