"""Unit tests for subject-level accuracy helpers (no training involved).

Run with:  conda run -n ocd-symphony python -m pytest tests/test_subject_accuracy.py -q
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from OCD_symphony.model.finetuning import _subject_ids_and_labels, subject_accuracy


def test_mean_probability_decides_not_majority():
    # Three samples of subject A (label 0). Majority vote says class 1 (2 vs 1),
    # but the mean probability favours class 0 (0.567 vs 0.433).
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.4, 0.6]])
    acc, n = subject_accuracy(probs, [0, 0, 0], ["A", "A", "A"])
    assert acc == 1.0 and n == 1


def test_subject_accuracy_over_several_subjects():
    probs = np.array([
        [0.8, 0.2], [0.7, 0.3],   # subject A, label 0 -> correct
        [0.1, 0.9], [0.4, 0.6],   # subject B, label 1 -> correct
        [0.6, 0.4], [0.9, 0.1],   # subject C, label 1 -> wrong
    ])
    labels = [0, 0, 1, 1, 1, 1]
    subj = ["A", "A", "B", "B", "C", "C"]
    acc, n = subject_accuracy(probs, labels, subj)
    assert n == 3
    assert acc == pytest.approx(2 / 3)


def test_single_sample_subjects_equal_sample_accuracy():
    rng = np.random.default_rng(0)
    probs = rng.random((20, 2))
    labels = rng.integers(0, 2, 20)
    subj = np.arange(20)
    acc, n = subject_accuracy(probs, labels, subj)
    sample_acc = (probs.argmax(1) == labels).mean()
    assert n == 20
    assert acc == pytest.approx(sample_acc)


def test_empty_input_is_nan_not_zero():
    acc, n = subject_accuracy(np.zeros((0, 2)), [], [])
    assert math.isnan(acc) and n == 0


def test_inconsistent_subject_labels_raise():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="different labels"):
        subject_accuracy(probs, [0, 1], ["A", "A"])


def test_tie_goes_to_lower_class_index():
    probs = np.array([[0.5, 0.5]])
    acc, _ = subject_accuracy(probs, [0], ["A"])
    assert acc == 1.0  # argmax of a tie is the lower index (0)


def _fake_dataset(n=10):
    subjs = np.array([f"s{i // 2}" for i in range(n)])        # two samples per subject
    labels = torch.tensor([i % 2 for i in range(n)])
    return SimpleNamespace(subjs=subjs, encoded_labels=labels, labels_tensor=labels)


def test_subject_mapping_through_nested_subset():
    base = _fake_dataset(10)
    inner = Subset(base, [9, 8, 7, 6, 5, 4])     # local position p -> base index
    outer = Subset(inner, [0, 2, 4])              # -> inner positions 0,2,4 -> base 9,7,5
    subj, enc = _subject_ids_and_labels(outer, [0, 1, 2])
    assert list(subj) == ["s4", "s3", "s2"]
    assert list(enc) == [1, 1, 1]                 # labels of base 9, 7, 5


def test_subject_mapping_plain_dataset():
    base = _fake_dataset(6)
    subj, enc = _subject_ids_and_labels(base, [0, 3, 5])
    assert list(subj) == ["s0", "s1", "s2"]
    assert list(enc) == [0, 1, 1]
