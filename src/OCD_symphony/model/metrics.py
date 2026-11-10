"""Pure metric and bookkeeping helpers for fine-tuning and DAPT.

Moved out of ``finetuning.py`` (Phase 2 of the refactor plan). ``finetuning.py``
re-exports every name here, so existing imports keep working.
"""
import numpy as np
import torch
from torch.utils.data import Subset


def _root_indices(ds):
    """Return (root_positions, root_dataset): positions of ``ds`` within its base dataset."""
    pos = np.arange(len(ds))
    base = ds
    while isinstance(base, Subset):
        pos = np.asarray(base.indices, dtype=np.int64)[pos]
        base = base.dataset
    return pos, base


def _subject_ids_and_labels(dataset, positions):
    """Return (subject_ids, targets) of the samples at ``positions`` of ``dataset``.

    ``targets`` is what the model receives: class indices, or raw ages for regression.
    ``dataset`` may be a (nested) ``Subset``; positions are local to it.
    """
    pos = np.asarray(positions, dtype=np.int64)
    base = dataset
    while isinstance(base, Subset):
        pos = np.asarray(base.indices, dtype=np.int64)[pos]
        base = base.dataset
    return np.asarray(base.subjs)[pos], base.labels_tensor[torch.as_tensor(pos)].numpy()


def _regression_metrics(targets, preds):
    """MAE, R² and Pearson r of predicted vs true values (NaN when undefined)."""
    t = np.asarray(targets, dtype=np.float64)
    p = np.asarray(preds, dtype=np.float64)
    if t.size == 0:
        return {"mae": float("nan"), "r2": float("nan"), "r": float("nan")}
    mae = float(np.mean(np.abs(p - t)))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((t - p) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(t, p)[0, 1]) if t.size > 1 and np.std(t) > 0 and np.std(p) > 0 else float("nan")
    return {"mae": mae, "r2": r2, "r": r}


def subject_regression_mae(preds, targets, subj_ids):
    """Subject-level MAE: mean prediction over a subject's windows vs its (constant) target."""
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    subj_ids = np.asarray(subj_ids)
    if subj_ids.size == 0:
        return float("nan"), 0
    uniq, inv = np.unique(subj_ids, return_inverse=True)
    sums = np.bincount(inv, weights=preds)
    counts = np.bincount(inv)
    subj_pred = sums / counts
    subj_true = np.zeros(len(uniq))
    subj_true[inv] = targets
    return float(np.mean(np.abs(subj_pred - subj_true))), int(len(uniq))


def subject_accuracy(probs, labels, subj_ids):
    """Subject-level accuracy from per-sample class probabilities.

    Each subject's samples are averaged (mean softmax probability) and the argmax is the
    subject prediction (ties go to the lower class index). Returns ``(accuracy, n_subjects)``;
    accuracy is NaN when there are no samples.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    subj_ids = np.asarray(subj_ids)
    if subj_ids.size == 0:
        return float("nan"), 0
    uniq, inv = np.unique(subj_ids, return_inverse=True)
    sums = np.zeros((len(uniq), probs.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, probs)
    mean_probs = sums / np.bincount(inv)[:, None]
    subj_label = np.full(len(uniq), -1, dtype=np.int64)
    subj_label[inv] = labels
    if not np.array_equal(labels, subj_label[inv]):
        raise ValueError("samples of one subject carry different labels")
    pred = mean_probs.argmax(axis=1)
    return float((pred == subj_label).mean()), int(len(uniq))


class _EarlyStopper:
    """Track the best validation loss and signal when training should stop.

    ``patience <= 0`` disables stopping (the best epoch is still tracked).
    A non-finite loss never counts as an improvement but does count toward patience.
    """

    def __init__(self, patience: int = 0):
        self.patience = int(patience)
        self.best_loss = float("inf")
        self.best_epoch = None
        self._epochs_since_best = 0

    def update(self, val_loss, epoch: int) -> bool:
        """Record one epoch. Returns True when ``val_loss`` is a new best."""
        if val_loss is None or not np.isfinite(val_loss):
            self._epochs_since_best += 1
            return False
        if val_loss < self.best_loss:
            self.best_loss = float(val_loss)
            self.best_epoch = epoch
            self._epochs_since_best = 0
            return True
        self._epochs_since_best += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self._epochs_since_best >= self.patience
