"""
Byte Pair Encoding (BPE) Tokenizer Implementation

BPE is a subword tokenization algorithm that iteratively merges the most frequent
pairs of bytes/characters to build a vocabulary of subword units.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.

    This tokenizer learns a vocabulary of subword units by iteratively merging
    the most frequent pairs of tokens in the training corpus.

    Attributes:
        vocab_size: Target vocabulary size
        vocab: Mapping from token string to token ID
        inverse_vocab: Mapping from token ID to token string
        merges: List of merge rules as (token1, token2) tuples
        special_tokens: Dictionary of special tokens (e.g., <PAD>, <UNK>)
    """

    # Pre-tokenization pattern (similar to GPT-2)
    # Splits on whitespace, punctuation, and keeps contractions together
    # Uses standard re module compatible pattern
    PAT = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""",
        re.IGNORECASE
    )

    def __init__(
        self,
        vocab_size: int = 1000,
        special_tokens: dict[str, int] | None = None
    ):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (including special tokens)
            special_tokens: Optional dictionary of special tokens to their IDs
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }

        # Initialize vocabulary with special tokens
        self.vocab: dict[str, int] = dict(self.special_tokens)
        self.inverse_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Merge rules learned during training
        self.merges: list[tuple[str, str]] = []

        # Cache for encoding efficiency
        self._merge_cache: dict[tuple[str, str], str] = {}

        # Track if tokenizer has been trained
        self._trained = False

    def _pre_tokenize(self, text: str) -> list[str]:
        """
        Split text into initial tokens before BPE.

        Args:
            text: Input text to pre-tokenize

        Returns:
            List of pre-tokenized strings
        """
        tokens = self.PAT.findall(text)
        return tokens if tokens else [text] if text else []

    def _get_pairs(self, tokens: list[str]) -> Counter[tuple[str, str]]:
        """
        Get frequency counts of adjacent token pairs.

        Args:
            tokens: List of tokens

        Returns:
            Counter of (token1, token2) pairs and their frequencies
        """
        pairs: Counter[tuple[str, str]] = Counter()
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_pair(
        self,
        tokens: list[str],
        pair: tuple[str, str]
    ) -> list[str]:
        """
        Merge all occurrences of a pair in the token list.

        Args:
            tokens: List of tokens
            pair: The (token1, token2) pair to merge

        Returns:
            New list with merged tokens
        """
        merged = pair[0] + pair[1]
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def train(
        self,
        texts: list[str] | str,
        verbose: bool = False
    ) -> None:
        """
        Train the BPE tokenizer on a corpus.

        Args:
            texts: Training corpus (single string or list of strings)
            verbose: Whether to print progress information
        """
        if isinstance(texts, str):
            texts = [texts]

        # Pre-tokenize all texts and count word frequencies
        word_freqs: Counter[tuple[str, ...]] = Counter()

        for text in texts:
            for word in self._pre_tokenize(text):
                # Convert word to tuple of characters
                char_tuple = tuple(word)
                word_freqs[char_tuple] += 1

        # Initialize vocabulary with all unique characters
        all_chars: set[str] = set()
        for word_tuple in word_freqs:
            all_chars.update(word_tuple)

        next_id = len(self.vocab)
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = next_id
                self.inverse_vocab[next_id] = char
                next_id += 1

        # Convert word_freqs to use list format for merging
        # Format: {word_as_tuple: frequency}
        splits: dict[tuple[str, ...], int] = dict(word_freqs)

        print("splits: ", splits)

        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        print(f"vocab size: {self.vocab_size}, len vocab: {len(self.vocab)}, merges: {num_merges}")

        for i in range(num_merges):
            # Count all pairs across the corpus
            pair_freqs: Counter[tuple[str, str]] = Counter()

            for word_tuple, freq in splits.items():
                print('word_tuple: ', word_tuple, 'len: ', len(word_tuple))
                for j in range(len(word_tuple) - 1):
                    pair = (word_tuple[j], word_tuple[j + 1])
                    pair_freqs[pair] += freq
                    print("pair: ", pair)

            if not pair_freqs:
                if verbose:
                    print(f"No more pairs to merge at iteration {i}")
                break

            # Find the most frequent pair
            best_pair = pair_freqs.most_common(1)[0][0]
            best_freq = pair_freqs[best_pair]

            if verbose and (i + 1) % 1 == 0:
                print(
                    f"Merge {i + 1}/{num_merges}: "
                    f"{best_pair} -> {''.join(best_pair)} (freq: {best_freq})"
                )

            # Record the merge rule
            self.merges.append(best_pair)

            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = next_id
                self.inverse_vocab[next_id] = merged_token
                next_id += 1

            # Apply merge to all words
            new_splits: dict[tuple[str, ...], int] = {}
            for word_tuple, freq in splits.items():
                # Convert to list, merge, convert back to tuple
                word_list = list(word_tuple)
                merged_list = self._merge_pair(word_list, best_pair)
                new_tuple = tuple(merged_list)
                new_splits[new_tuple] = new_splits.get(new_tuple, 0) + freq

            splits = new_splits

        self._trained = True

        if verbose:
            print(f"Training complete. Vocabulary size: {len(self.vocab)}")

    def _tokenize_word(self, word: str) -> list[str]:
        """
        Tokenize a single word using learned merge rules.

        Args:
            word: Word to tokenize

        Returns:
            List of subword tokens
        """
        if not word:
            return []

        # Start with characters
        tokens = list(word)

        # Apply merge rules in order
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
            if len(tokens) == 1:
                break

        return tokens

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False
    ) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if not self._trained and len(self.vocab) <= len(self.special_tokens):
            raise ValueError(
                "Tokenizer has not been trained. Call train() first."
            )

        token_ids: list[int] = []

        if add_special_tokens:
            token_ids.append(self.special_tokens["<BOS>"])

        # Pre-tokenize and then apply BPE
        for word in self._pre_tokenize(text):
            subwords = self._tokenize_word(word)
            for subword in subwords:
                if subword in self.vocab:
                    token_ids.append(self.vocab[subword])
                else:
                    # Handle unknown tokens
                    token_ids.append(self.special_tokens["<UNK>"])

        if add_special_tokens:
            token_ids.append(self.special_tokens["<EOS>"])

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        special_ids = set(self.special_tokens.values())
        tokens: list[str] = []

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
            else:
                # Unknown token ID
                if not skip_special_tokens:
                    tokens.append("<UNK>")

        return "".join(tokens)

    def get_vocab(self) -> dict[str, int]:
        """
        Get the vocabulary dictionary.

        Returns:
            Dictionary mapping tokens to IDs
        """
        return dict(self.vocab)

    def get_vocab_size(self) -> int:
        """
        Get the current vocabulary size.

        Returns:
            Number of tokens in vocabulary
        """
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer to a JSON file.

        Args:
            path: Path to save the tokenizer
        """
        path = Path(path)

        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """
        Load tokenizer from a JSON file.

        Args:
            path: Path to the saved tokenizer

        Returns:
            Loaded BPETokenizer instance
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"],
        )

        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer._trained = True

        return tokenizer

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"current_vocab={len(self.vocab)}, "
            f"merges={len(self.merges)})"
        )


def tokenize(text: str, tokenizer: BPETokenizer) -> list[int]:
    """
    Convenience function to tokenize text.

    Args:
        text: Text to tokenize
        tokenizer: Trained BPE tokenizer

    Returns:
        List of token IDs
    """
    return tokenizer.encode(text)


def detokenize(token_ids: list[int], tokenizer: BPETokenizer) -> str:
    """
    Convenience function to detokenize token IDs.

    Args:
        token_ids: List of token IDs
        tokenizer: Trained BPE tokenizer

    Returns:
        Decoded text
    """
    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    # Example usage
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "The dog barks at the fox.",
        "Quick foxes are brown.",
        "Lazy dogs sleep all day.",
        "The brown dog is lazy.",
    ]

    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus, verbose=True)

    # Test encoding and decoding
    test_text = "The quick fox"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {len(tokenizer)}")

    tokenizer.save('./test.json')
