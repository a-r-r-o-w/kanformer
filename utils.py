import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn


def get_summary(module: nn.Module):
    trainable_parameters = sum(
        [p.numel() for p in module.parameters() if p.requires_grad]
    )
    untrainable_parameters = sum(
        [p.numel() for p in module.parameters() if not p.requires_grad]
    )

    dtype = next(module.parameters()).dtype
    size = 2 if dtype == torch.float16 else 4
    memory = size * (trainable_parameters + untrainable_parameters) / 1024 / 1024

    return {
        "trainable": trainable_parameters,
        "untrainable": untrainable_parameters,
        "memory": memory,
    }


def initialize_weights(module: nn.Module, method: str, **kwargs) -> None:
    if method == "kaiming_uniform":
        m = nn.init.kaiming_uniform_
    elif method == "kaiming_normal":
        m = nn.init.kaiming_normal_
    elif method == "xavier_uniform":
        m = nn.init.xavier_uniform_
    elif method == "xavier_normal":
        m = nn.init.xavier_normal_
    elif method == "uniform":
        m = nn.init.uniform_
    elif method == "normal":
        m = nn.init.normal_

    def init(x):
        if hasattr(x, "weight") and x.weight.dim() > 1:
            m(x.weight.data, **kwargs)

    module.apply(init)


def pad_sequence(
    sequences: List[torch.Tensor], padding_value: int, max_length: int
) -> torch.Tensor:
    padded_sequences = torch.full(
        (len(sequences), max_length), padding_value, dtype=torch.long
    )
    for i, sequence in enumerate(sequences):
        padded_sequences[i, : sequence.size(0)] = sequence
    return padded_sequences


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    en_pad_token_id: int,
    de_pad_token_id: int,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    en_tensors, de_tensors = zip(*batch)
    en_tensors = pad_sequence(
        en_tensors, padding_value=en_pad_token_id, max_length=max_length
    )
    de_tensors = pad_sequence(
        de_tensors, padding_value=de_pad_token_id, max_length=max_length
    )
    return en_tensors, de_tensors


def get_ngrams(sentence: List[str], n: int) -> Dict[Tuple[str], int]:
    ngram_counts = {}
    for i in range(1, n + 1):
        for j in range(len(sentence) - i + 1):
            ngram = tuple(sentence[j : j + i])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts


def _bleu_score(
    translation: Union[str, List[str]],
    references: Union[List[str], List[List[str]]],
    n_gram: int = 4,
) -> float:
    if isinstance(translation, str):
        translation = [translation]
    if isinstance(references[0], str):
        references = [references]
    if len(translation) != len(references):
        raise ValueError("The number of translations and references do not match")

    for translation_, references_ in zip(translation, references):
        translation_ = translation_.split()
        references_ = [ref.split() for ref in references_]
        len_translation = len(translation_)
        len_reference = min([len(ref) for ref in references_])
        input_len, target_len = 0, 0

        input_len += len_translation
        target_len += len_reference

        translation_ngram_counter = get_ngrams(translation_, n_gram)
        reference_ngram_counter = {}
        for ref in references_:
            reference_ngram_counter_ = get_ngrams(ref, n_gram)
            for ngram in reference_ngram_counter_:
                reference_ngram_counter[ngram] = max(
                    reference_ngram_counter.get(ngram, 0),
                    reference_ngram_counter_[ngram],
                )
        overlap = {
            ngram: min(
                translation_ngram_counter.get(ngram),
                reference_ngram_counter.get(ngram, 0),
            )
            for ngram in translation_ngram_counter
        }

        matches_by_order = [0] * n_gram
        possible_matches_by_order = [0] * n_gram
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for i in range(n_gram):
            if i < len_translation:
                possible_matches_by_order[i] += len_translation - i

    if min(possible_matches_by_order) == 0:
        raise ValueError("Input too small to find n-gram matches")

    weights = [1 / n_gram] * n_gram
    precision = [m / p for m, p in zip(matches_by_order, possible_matches_by_order)]
    geometric_mean = math.exp(
        sum(
            [
                w * (math.log(p) if p > 0 else -float("inf"))
                for w, p in zip(weights, precision)
            ]
        )
    )
    brevity_penalty = (
        1 if input_len > target_len else math.exp(1 - target_len / input_len)
    )

    return brevity_penalty * geometric_mean


def bleu_score(
    translation: Union[str, List[str]],
    references: Union[List[str], List[List[str]]],
    n_gram: int = 4,
) -> float:
    try:
        score = _bleu_score(translation, references, n_gram)
    except Exception as e:
        print(f"Exception: {e}")
        score = 0.0
    return score
