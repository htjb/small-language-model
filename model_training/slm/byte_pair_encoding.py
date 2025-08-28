import re
from collections import Counter

import numpy as np
import torch


class bpe:
    def __init__(self, files, num_merges=1000):
        text = []
        for f in files:
            with open(f, "r") as file:
                text.append(file.readlines())
        text = np.concatenate(text)

        tokenized = [
            re.findall(r"\w+|[^\w\s]", line, flags=re.UNICODE) for line in text
        ]

        words = [list(word) for word in np.concatenate(tokenized)]

        mergers = 0
        merger_rules = {}
        while mergers < num_merges:
            pairs = Counter()
            for word in words:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] += 1
            most_common = pairs.most_common()[0]
            merger_rules[most_common[0]] = mergers

            for i in range(len(words)):
                j = 0
                while j < len(words[i]) - 1:
                    if (words[i][j], words[i][j + 1]) == most_common[0]:
                        words[i] = (
                            words[i][:j]
                            + ["".join(most_common[0])]
                            + words[i][j + 2 :]
                        )
                        j += 1
                    else:
                        j += 1
            mergers += 1

        processed_words = []
        for w in words:
            processed_words.append(w)
            if w in [".", "!", "?"]:
                processed_words.append("EOS")
        processed_words.append("EOS")

        vocab = np.unique(np.concatenate(words)).tolist()
        vocab.append("UNK")
        vocab.append("EOS")
        vocab.append(" ")

        self.vocab = vocab
        self.merger_rules = merger_rules

        self.word_to_index = {word: i + 1 for i, word in enumerate(vocab)}
        self.index_to_word = {i + 1: word for i, word in enumerate(vocab)}

        codified = []
        for i, word in enumerate(processed_words):
            for subword in word:
                if subword in [".", "!", "?"]:
                    codified.pop()
                    codified.append(subword)
                    codified.append("EOS")
                elif subword in self.vocab:
                    codified.append(subword)
                else:
                    codified.append("UNK")
            if i < len(tokenized) - 1:
                codified.append(" ")

        counter = Counter(codified)
        self.freqs = torch.tensor(
            [counter.get(word, 0) for word in vocab], dtype=torch.float
        )

    def codify(self, line):
        line = line.strip()
        words = re.findall(r"\w+|[^\w\s]", line)
        tokenized = [list(word) for word in words]

        for i in range(len(tokenized)):
            j = 0
            while j < len(tokenized[i]) - 1:
                pair = (tokenized[i][j], tokenized[i][j + 1])
                if pair in self.merger_rules:
                    tokenized[i] = (
                        tokenized[i][:j]
                        + ["".join(pair)]
                        + tokenized[i][j + 2 :]
                    )
                    j = 0
                else:
                    j += 1

        codified = []
        for i, word in enumerate(tokenized):
            for subword in word:
                if subword in [".", "!", "?"]:
                    codified.pop()
                    codified.append(subword)
                    codified.append("EOS")
                elif subword in self.vocab:
                    codified.append(subword)
                else:
                    codified.append("UNK")
            if i < len(tokenized) - 1:
                codified.append(" ")

        indices = torch.tensor(
            [
                self.word_to_index.get(tok, self.word_to_index["UNK"])
                for tok in codified
            ]
        )

        return indices
