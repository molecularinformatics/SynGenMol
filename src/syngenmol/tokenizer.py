"""Tokenizer utilities for building-block-token trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

DEFAULT_SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>")


class BuildingBlockTokenizer:
    """A word-level tokenizer with one vocabulary item per building block.

    Building-block identifiers must not contain whitespace because a trajectory is
    represented as a whitespace-separated sequence of identifiers.  Special tokens
    are always appended after building-block tokens in a deterministic order.
    """

    def __init__(
        self,
        tokens: Iterable[str] | None = None,
        special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    ) -> None:
        special_tokens = tuple(special_tokens)
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("special_tokens must be unique")

        vocabulary: dict[str, int] = {}
        for token in tokens or ():
            token = str(token)
            if not token:
                raise ValueError("Building-block identifiers must be nonempty")
            if any(character.isspace() for character in token):
                raise ValueError(
                    f"Building-block identifier {token!r} contains whitespace; "
                    "identifiers are trajectory tokens and must be whitespace-free."
                )
            if token in special_tokens:
                raise ValueError(f"Building-block identifier collides with special token: {token!r}")
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

        for token in special_tokens:
            vocabulary.setdefault(token, len(vocabulary))

        self._special_tokens = special_tokens
        self._vocab = vocabulary
        self._fast_tokenizer = self._make_fast_tokenizer(vocabulary, special_tokens)

    @staticmethod
    def _make_fast_tokenizer(
        vocabulary: Mapping[str, int], special_tokens: Sequence[str]
    ) -> PreTrainedTokenizerFast:
        tokenizer = Tokenizer(WordLevel(dict(vocabulary), unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>" if "<s>" in special_tokens else None,
            eos_token="</s>" if "</s>" in special_tokens else None,
            unk_token="<unk>" if "<unk>" in special_tokens else None,
            pad_token="<pad>" if "<pad>" in special_tokens else None,
        )

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """The Hugging Face tokenizer used by the model and PPO implementation."""
        return self._fast_tokenizer

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def token_to_id(self) -> dict[str, int]:
        return self.vocab

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self._vocab.items()}

    @property
    def special_tokens(self) -> tuple[str, ...]:
        return self._special_tokens

    @property
    def special_token_ids(self) -> set[int]:
        return {
            token_id
            for token_id in (
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            )
            if token_id is not None
        }

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
        (directory / "vocab.json").write_text(
            json.dumps(self._vocab, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "special_tokens.json").write_text(
            json.dumps(list(self._special_tokens), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "BuildingBlockTokenizer":
        directory = Path(directory)
        vocab_path = directory / "vocab.json"
        if not vocab_path.is_file():
            raise FileNotFoundError(f"Tokenizer vocabulary not found: {vocab_path}")

        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        specials_path = directory / "special_tokens.json"
        if specials_path.is_file():
            special_tokens = tuple(json.loads(specials_path.read_text(encoding="utf-8")))
        else:
            special_tokens = tuple(token for token in DEFAULT_SPECIAL_TOKENS if token in vocabulary)

        instance = cls(tokens=(), special_tokens=special_tokens)
        instance._vocab = {str(token): int(index) for token, index in vocabulary.items()}
        instance._fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(directory)
        return instance


# Compatibility alias for legacy notebook and script imports. New code should use
# BuildingBlockTokenizer explicitly.
CustomTokenizer = BuildingBlockTokenizer
