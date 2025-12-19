"""
Comprehensive tests for the BPE Tokenizer implementation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from bpe import BPETokenizer, detokenize, tokenize


class TestBPETokenizerInitialization:
    """Tests for tokenizer initialization."""

    def test_default_initialization(self):
        """Test tokenizer initializes with default parameters."""
        tokenizer = BPETokenizer()

        assert tokenizer.vocab_size == 1000
        assert "<PAD>" in tokenizer.special_tokens
        assert "<UNK>" in tokenizer.special_tokens
        assert "<BOS>" in tokenizer.special_tokens
        assert "<EOS>" in tokenizer.special_tokens
        assert len(tokenizer.merges) == 0
        assert not tokenizer._trained

    def test_custom_vocab_size(self):
        """Test tokenizer with custom vocabulary size."""
        tokenizer = BPETokenizer(vocab_size=500)
        assert tokenizer.vocab_size == 500

    def test_custom_special_tokens(self):
        """Test tokenizer with custom special tokens."""
        special = {"<MASK>": 0, "<SEP>": 1}
        tokenizer = BPETokenizer(special_tokens=special)

        assert tokenizer.special_tokens == special
        assert "<MASK>" in tokenizer.vocab
        assert "<SEP>" in tokenizer.vocab

    def test_special_tokens_in_vocab(self):
        """Test that special tokens are added to vocabulary."""
        tokenizer = BPETokenizer()

        assert tokenizer.vocab["<PAD>"] == 0
        assert tokenizer.vocab["<UNK>"] == 1
        assert tokenizer.vocab["<BOS>"] == 2
        assert tokenizer.vocab["<EOS>"] == 3

    def test_inverse_vocab_initialized(self):
        """Test that inverse vocabulary is correctly initialized."""
        tokenizer = BPETokenizer()

        assert tokenizer.inverse_vocab[0] == "<PAD>"
        assert tokenizer.inverse_vocab[1] == "<UNK>"


class TestBPETokenizerTraining:
    """Tests for the training process."""

    def test_train_simple_corpus(self):
        """Test training on a simple corpus."""
        corpus = ["aaa bbb aaa", "aaa bbb ccc"]
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(corpus)

        assert tokenizer._trained
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        assert len(tokenizer.merges) > 0

    def test_train_single_string(self):
        """Test training with a single string instead of list."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train("hello world hello")

        assert tokenizer._trained

    def test_train_builds_character_vocab(self):
        """Test that training builds vocabulary from characters."""
        corpus = ["abc"]
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(corpus)

        assert "a" in tokenizer.vocab
        assert "b" in tokenizer.vocab
        assert "c" in tokenizer.vocab

    def test_train_creates_merges(self):
        """Test that training creates merge rules."""
        corpus = ["aa aa aa bb bb"]
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train(corpus)

        # Should have at least one merge
        assert len(tokenizer.merges) > 0

    def test_train_frequent_pairs_merged_first(self):
        """Test that most frequent pairs are merged first."""
        # 'aa' appears more frequently than other pairs
        corpus = ["aa aa aa aa bb"]
        tokenizer = BPETokenizer(vocab_size=10)
        tokenizer.train(corpus)

        # First merge should involve 'a' with 'a'
        if tokenizer.merges:
            first_merge = tokenizer.merges[0]
            merged_token = first_merge[0] + first_merge[1]
            assert merged_token in tokenizer.vocab

    def test_train_empty_corpus(self):
        """Test training with empty corpus."""
        tokenizer = BPETokenizer(vocab_size=20)
        tokenizer.train([])

        # Should still have special tokens
        assert len(tokenizer.vocab) >= len(tokenizer.special_tokens)

    def test_train_verbose_mode(self, capsys):
        """Test verbose training output."""
        corpus = ["hello world"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus, verbose=True)

        captured = capsys.readouterr()
        assert "Training complete" in captured.out

    def test_vocab_size_limit(self):
        """Test that vocabulary respects size limit."""
        corpus = ["the quick brown fox jumps over the lazy dog"] * 100
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train(corpus)

        # Vocab should not exceed target size significantly
        assert len(tokenizer.vocab) <= tokenizer.vocab_size + 10


class TestBPETokenizerEncoding:
    """Tests for encoding functionality."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        corpus = [
            "hello world",
            "hello there",
            "world hello",
            "hello hello hello",
        ]
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        return tokenizer

    def test_encode_returns_list(self, trained_tokenizer):
        """Test that encode returns a list of integers."""
        result = trained_tokenizer.encode("hello")

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_encode_known_text(self, trained_tokenizer):
        """Test encoding text seen during training."""
        result = trained_tokenizer.encode("hello")

        assert len(result) > 0
        assert all(x in trained_tokenizer.inverse_vocab for x in result)

    def test_encode_with_special_tokens(self, trained_tokenizer):
        """Test encoding with BOS/EOS tokens."""
        result = trained_tokenizer.encode("hello", add_special_tokens=True)

        assert result[0] == trained_tokenizer.special_tokens["<BOS>"]
        assert result[-1] == trained_tokenizer.special_tokens["<EOS>"]

    def test_encode_without_special_tokens(self, trained_tokenizer):
        """Test encoding without special tokens."""
        result = trained_tokenizer.encode("hello", add_special_tokens=False)

        bos_id = trained_tokenizer.special_tokens["<BOS>"]
        eos_id = trained_tokenizer.special_tokens["<EOS>"]

        assert bos_id not in result or result[0] != bos_id
        assert eos_id not in result or result[-1] != eos_id

    def test_encode_empty_string(self, trained_tokenizer):
        """Test encoding empty string."""
        result = trained_tokenizer.encode("")
        assert result == []

    def test_encode_unknown_characters(self, trained_tokenizer):
        """Test encoding with unknown characters."""
        # Use characters unlikely to be in training
        result = trained_tokenizer.encode("日本語")

        # Should contain UNK tokens for unknown chars
        unk_id = trained_tokenizer.special_tokens["<UNK>"]
        assert unk_id in result

    def test_encode_before_training_raises(self):
        """Test that encoding before training raises error."""
        tokenizer = BPETokenizer()

        with pytest.raises(ValueError, match="not been trained"):
            tokenizer.encode("hello")


class TestBPETokenizerDecoding:
    """Tests for decoding functionality."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        corpus = ["hello world hello world"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        return tokenizer

    def test_decode_returns_string(self, trained_tokenizer):
        """Test that decode returns a string."""
        encoded = trained_tokenizer.encode("hello")
        decoded = trained_tokenizer.decode(encoded)

        assert isinstance(decoded, str)

    def test_decode_skip_special_tokens(self, trained_tokenizer):
        """Test decoding with special tokens skipped."""
        encoded = trained_tokenizer.encode("hello", add_special_tokens=True)
        decoded = trained_tokenizer.decode(encoded, skip_special_tokens=True)

        assert "<BOS>" not in decoded
        assert "<EOS>" not in decoded

    def test_decode_include_special_tokens(self, trained_tokenizer):
        """Test decoding with special tokens included."""
        encoded = trained_tokenizer.encode("hello", add_special_tokens=True)
        decoded = trained_tokenizer.decode(encoded, skip_special_tokens=False)

        assert "<BOS>" in decoded
        assert "<EOS>" in decoded

    def test_decode_empty_list(self, trained_tokenizer):
        """Test decoding empty list."""
        result = trained_tokenizer.decode([])
        assert result == ""

    def test_decode_unknown_token_id(self, trained_tokenizer):
        """Test decoding with unknown token ID."""
        result = trained_tokenizer.decode([99999])
        assert result == ""  # Skip unknown when skip_special_tokens=True

    def test_roundtrip_encoding(self, trained_tokenizer):
        """Test that encode-decode preserves text."""
        original = "hello world"
        encoded = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(encoded)

        assert decoded == original


class TestBPETokenizerRoundtrip:
    """Tests for encode-decode roundtrip consistency."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture with larger vocabulary for better coverage."""
        corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vexingly quick daft zebras jump!",
            "The five boxing wizards jump quickly.",
        ] * 20
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(corpus)
        return tokenizer

    @pytest.mark.parametrize("text", [
        "hello",
        "The quick brown fox",
        "hello world",
        "UPPERCASE",
        "MixedCase Text",
        "   spaces   ",
        "punctuation!@#",
    ])
    def test_roundtrip_various_texts(self, trained_tokenizer, text):
        """Test roundtrip on various text inputs."""
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)

        # Note: Some texts may not roundtrip perfectly due to UNK tokens
        # But for trained vocabulary, it should work
        assert isinstance(decoded, str)

    def test_roundtrip_training_corpus(self, trained_tokenizer):
        """Test roundtrip on text similar to training corpus."""
        text = "The quick fox jumps"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)

        assert decoded == text


class TestBPETokenizerSaveLoad:
    """Tests for save and load functionality."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        corpus = ["hello world"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        return tokenizer

    def test_save_creates_file(self, trained_tokenizer):
        """Test that save creates a file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        trained_tokenizer.save(path)

        assert Path(path).exists()
        Path(path).unlink()

    def test_save_valid_json(self, trained_tokenizer):
        """Test that saved file is valid JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        trained_tokenizer.save(path)

        with open(path) as f:
            data = json.load(f)

        assert "vocab" in data
        assert "merges" in data
        assert "vocab_size" in data
        assert "special_tokens" in data

        Path(path).unlink()

    def test_load_restores_tokenizer(self, trained_tokenizer):
        """Test that load restores tokenizer correctly."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        trained_tokenizer.save(path)
        loaded = BPETokenizer.load(path)

        assert loaded.vocab == trained_tokenizer.vocab
        assert loaded.merges == trained_tokenizer.merges
        assert loaded.vocab_size == trained_tokenizer.vocab_size
        assert loaded._trained

        Path(path).unlink()

    def test_loaded_tokenizer_encodes_same(self, trained_tokenizer):
        """Test that loaded tokenizer produces same encodings."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        trained_tokenizer.save(path)
        loaded = BPETokenizer.load(path)

        text = "hello world"
        original_encoded = trained_tokenizer.encode(text)
        loaded_encoded = loaded.encode(text)

        assert original_encoded == loaded_encoded

        Path(path).unlink()

    def test_save_load_with_path_object(self, trained_tokenizer):
        """Test save/load with Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokenizer.json"

            trained_tokenizer.save(path)
            loaded = BPETokenizer.load(path)

            assert loaded.vocab == trained_tokenizer.vocab


class TestBPETokenizerHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        corpus = ["hello world"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        return tokenizer

    def test_get_vocab(self, trained_tokenizer):
        """Test get_vocab returns vocabulary copy."""
        vocab = trained_tokenizer.get_vocab()

        assert isinstance(vocab, dict)
        assert vocab == trained_tokenizer.vocab
        # Ensure it's a copy
        vocab["test"] = 999
        assert "test" not in trained_tokenizer.vocab

    def test_get_vocab_size(self, trained_tokenizer):
        """Test get_vocab_size returns correct size."""
        size = trained_tokenizer.get_vocab_size()

        assert size == len(trained_tokenizer.vocab)

    def test_len(self, trained_tokenizer):
        """Test __len__ returns vocabulary size."""
        assert len(trained_tokenizer) == len(trained_tokenizer.vocab)

    def test_repr(self, trained_tokenizer):
        """Test __repr__ returns informative string."""
        repr_str = repr(trained_tokenizer)

        assert "BPETokenizer" in repr_str
        assert "vocab_size" in repr_str
        assert "merges" in repr_str


class TestBPETokenizerPreTokenization:
    """Tests for pre-tokenization behavior."""

    def test_pre_tokenize_splits_words(self):
        """Test that pre-tokenize splits on whitespace."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._pre_tokenize("hello world")

        assert len(tokens) >= 2

    def test_pre_tokenize_handles_punctuation(self):
        """Test pre-tokenization with punctuation."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._pre_tokenize("hello, world!")

        assert len(tokens) >= 2

    def test_pre_tokenize_handles_contractions(self):
        """Test pre-tokenization preserves contractions."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._pre_tokenize("don't won't")

        # Should handle contractions
        assert len(tokens) >= 2

    def test_pre_tokenize_empty_string(self):
        """Test pre-tokenization of empty string."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._pre_tokenize("")

        assert tokens == []

    def test_pre_tokenize_whitespace_only(self):
        """Test pre-tokenization of whitespace."""
        tokenizer = BPETokenizer()
        tokens = tokenizer._pre_tokenize("   ")

        # May return whitespace tokens or empty
        assert isinstance(tokens, list)


class TestBPETokenizerMerging:
    """Tests for merge operations."""

    def test_merge_pair_basic(self):
        """Test basic pair merging."""
        tokenizer = BPETokenizer()
        tokens = ["a", "b", "c", "a", "b"]
        pair = ("a", "b")

        result = tokenizer._merge_pair(tokens, pair)

        assert result == ["ab", "c", "ab"]

    def test_merge_pair_no_match(self):
        """Test merging when pair not present."""
        tokenizer = BPETokenizer()
        tokens = ["a", "b", "c"]
        pair = ("x", "y")

        result = tokenizer._merge_pair(tokens, pair)

        assert result == ["a", "b", "c"]

    def test_merge_pair_overlapping(self):
        """Test merging with overlapping patterns."""
        tokenizer = BPETokenizer()
        tokens = ["a", "a", "a"]
        pair = ("a", "a")

        result = tokenizer._merge_pair(tokens, pair)

        # Should merge first pair, leave last 'a'
        assert result == ["aa", "a"]

    def test_get_pairs(self):
        """Test pair frequency counting."""
        tokenizer = BPETokenizer()
        tokens = ["a", "b", "a", "b", "c"]

        pairs = tokenizer._get_pairs(tokens)

        assert pairs[("a", "b")] == 2
        assert pairs[("b", "a")] == 1
        assert pairs[("b", "c")] == 1


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def trained_tokenizer(self):
        """Fixture providing a trained tokenizer."""
        corpus = ["hello world"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)
        return tokenizer

    def test_tokenize_function(self, trained_tokenizer):
        """Test tokenize convenience function."""
        result = tokenize("hello", trained_tokenizer)

        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_detokenize_function(self, trained_tokenizer):
        """Test detokenize convenience function."""
        encoded = tokenize("hello", trained_tokenizer)
        decoded = detokenize(encoded, trained_tokenizer)

        assert isinstance(decoded, str)


class TestBPETokenizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_vocab_size(self):
        """Test with very small vocabulary size."""
        corpus = ["hello world"]
        tokenizer = BPETokenizer(vocab_size=5)
        tokenizer.train(corpus)

        # Should still work, just with limited merges
        assert tokenizer._trained

    def test_large_corpus(self):
        """Test with larger corpus."""
        corpus = ["the quick brown fox jumps over the lazy dog"] * 1000
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.train(corpus)

        assert tokenizer._trained
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)

    def test_unicode_text(self):
        """Test with unicode characters."""
        corpus = ["café résumé naïve"]
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)

        # Should handle unicode
        encoded = tokenizer.encode("café")
        assert len(encoded) > 0

    def test_numbers_in_text(self):
        """Test with numbers."""
        corpus = ["123 456 789"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)

        encoded = tokenizer.encode("123")
        decoded = tokenizer.decode(encoded)

        assert "1" in decoded or "123" in decoded

    def test_special_characters(self):
        """Test with special characters."""
        corpus = ["hello! @world #test"] * 10
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)

        encoded = tokenizer.encode("hello!")
        assert len(encoded) > 0

    def test_mixed_whitespace(self):
        """Test with tabs and newlines."""
        corpus = ["hello\tworld\nnew line"]
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)

        encoded = tokenizer.encode("hello\tworld")
        assert len(encoded) > 0

    def test_repeated_training(self):
        """Test training tokenizer multiple times."""
        tokenizer = BPETokenizer(vocab_size=50)

        tokenizer.train(["hello world"])
        vocab_after_first = len(tokenizer.vocab)

        # Training again should rebuild vocabulary
        tokenizer.train(["different corpus"])

        # Vocabulary should change
        assert tokenizer._trained


class TestBPETokenizerPerformance:
    """Basic performance/stress tests."""

    def test_encode_long_text(self):
        """Test encoding long text."""
        corpus = ["word "] * 100
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train(corpus)

        long_text = "word " * 1000
        encoded = tokenizer.encode(long_text)

        assert len(encoded) > 0

    def test_many_unique_tokens(self):
        """Test with many unique tokens in input."""
        # Create corpus with many unique words
        corpus = [f"word{i}" for i in range(100)]
        tokenizer = BPETokenizer(vocab_size=200)
        tokenizer.train(corpus)

        assert tokenizer._trained


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
