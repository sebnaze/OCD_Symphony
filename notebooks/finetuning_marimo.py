import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full", app_title="BrainSymphony Fine-tuning")


@app.cell
def _():
    import os
    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        opts = [opt.strip() for opt in os.environ["PYTORCH_CUDA_ALLOC_CONF"].split(",") if not opt.strip().startswith("expandable_segments")]
        if opts:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(opts)
        else:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import importlib
    from types import SimpleNamespace

    # Force a reload so edits to finetuning.py are picked up without restarting
    # the marimo kernel (Python caches imported modules in sys.modules).
    import OCD_symphony.model.finetuning as _ft
    importlib.reload(_ft)

    from OCD_symphony.model.finetuning import (
        run_single_split,
        run_single_fold,
        run_cross_validation,
        run_dapt,
        clean_gpu_memory,
    )
    return (
        SimpleNamespace,
        clean_gpu_memory,
        mo,
        os,
        plt,
        run_cross_validation,
        run_dapt,
        run_single_fold,
        run_single_split,
    )


@app.cell(hide_code=True)
def _(mo):
    title_md = mo.md(
        "# 🧠 BrainSymphony Fine-Tuning Dashboard\n"
        "Interactive experimentation harness for fine-tuning the pretrained **Perceiver Classifier** on HCP fMRI embeddings."
    )
    title_md
    return


@app.cell
def _(mo):
    label_selector = mo.ui.dropdown(
        options=["gender", "age"],
        value="gender",
        label="Target Label",
    )
    mode_selector = mo.ui.radio(
        options=["Single Split", "Single Fold (CV)", "Full Cross-Validation (K-Fold)"],
        value="Single Split",
        label="Evaluation Mode",
    )
    fold_slider = mo.ui.slider(
        start=0,
        stop=4,
        value=0,
        step=1,
        label="Fold Index",
    )
    epochs_slider = mo.ui.slider(
        start=1,
        stop=200,
        value=10,
        step=1,
        label="Epochs",
    )
    lr_dropdown = mo.ui.dropdown(
        options=[1e-5, 5e-5, 1e-4, 3e-4, 1e-3],
        value=1e-4,
        label="Learning Rate",
    )
    batch_size_dropdown = mo.ui.dropdown(
        options=[4, 8, 16, 32],
        value=8,
        label="Batch Size",
    )
    subjs_input = mo.ui.dropdown(
        options={"All Subjects": None, "50 Subjects (Debug/Fast)": 50, "200 Subjects": 200, "500 Subjects": 500},
        value="50 Subjects (Debug/Fast)",
        label="Dataset Scope",
    )
    seed_input = mo.ui.number(
        start=0,
        stop=9999,
        value=42,
        label="Random Seed",
    )
    stage_selector = mo.ui.dropdown(
        options={
            "Stage 1: Linear Probe (Freeze Perceiver)": "stage1",
            "Stage 2: Fine-tune Perceiver Backbone (Unfreeze Perceiver)": "stage2",
            "Stage 3.1: DAPT — Reconstruction Pretrain on HCP-YA": "dapt",
            "Stage 3.2: Fine-tune from DAPT (encoders frozen / low-LR)": "stage3_2",
        },
        value="Stage 2: Fine-tune Perceiver Backbone (Unfreeze Perceiver)",
        label="Training Stage",
    )
    backbone_lr_dropdown = mo.ui.dropdown(
        options=[1e-6, 5e-6, 1e-5, 3e-5, 1e-4],
        value=1e-5,
        label="Perceiver LR",
    )
    wandb_checkbox = mo.ui.checkbox(
        value=False,
        label="Log run & model to Weights & Biases (W&B)",
    )
    pooling_selector = mo.ui.radio(
        options={"Mean Pooling": "mean", "Multi-Head Attention Pooling": "attention", "Flatten (Legacy)": "flatten"},
        value="Flatten (Legacy)",
        label="Latent Pooling",
    )
    dropout_slider = mo.ui.slider(
        start=0.0,
        stop=0.7,
        value=0.3,
        step=0.05,
        label="Head Dropout",
    )
    recon_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.0,
        step=0.05,
        label="Loss Balance α (0 = 100% Cls, 0.5 = 50/50, 1 = 100% Recon)",
    )
    # ── Stage 3 (DAPT) controls ─────────────────────────────────────────────
    dapt_mask_spatial_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.7,
        step=0.05,
        label="DAPT Mask Ratio — Spatial (Stage 3.1)",
    )
    dapt_mask_temporal_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.7,
        step=0.05,
        label="DAPT Mask Ratio — Temporal (Stage 3.1)",
    )
    dapt_mask_context_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.5,
        step=0.05,
        label="DAPT Mask Ratio — Context (Stage 3.1)",
    )
    dapt_weights_input = mo.ui.text(
        value="1,1,1,1",
        label="DAPT Loss Weights sp,tp,conv,perceiver (Stage 3.1)",
    )
    dapt_ckpt_name_input = mo.ui.text(
        value="",
        label="DAPT Output Name (Stage 3.1, optional)",
    )
    dapt_ckpt_input = mo.ui.text(
        value="",
        label="DAPT Checkpoint Path (Stage 3.2 input)",
    )
    encoder_lr_dropdown = mo.ui.dropdown(
        options=[0.0, 1e-7, 5e-7, 1e-6, 5e-6],
        value=0.0,
        label="Encoder LR (Stage 3.2; 0 = frozen)",
    )
    run_btn = mo.ui.run_button(
        label="🚀 Start Fine-tuning",
        kind="success",
    )
    return (
        backbone_lr_dropdown,
        batch_size_dropdown,
        dapt_ckpt_input,
        dapt_ckpt_name_input,
        dapt_mask_context_slider,
        dapt_mask_spatial_slider,
        dapt_mask_temporal_slider,
        dapt_weights_input,
        dropout_slider,
        encoder_lr_dropdown,
        epochs_slider,
        fold_slider,
        label_selector,
        lr_dropdown,
        mode_selector,
        pooling_selector,
        recon_slider,
        run_btn,
        seed_input,
        stage_selector,
        subjs_input,
        wandb_checkbox,
    )


@app.cell
def _(
    backbone_lr_dropdown,
    batch_size_dropdown,
    dapt_ckpt_input,
    dapt_ckpt_name_input,
    dapt_mask_context_slider,
    dapt_mask_spatial_slider,
    dapt_mask_temporal_slider,
    dapt_weights_input,
    dropout_slider,
    encoder_lr_dropdown,
    epochs_slider,
    fold_slider,
    label_selector,
    lr_dropdown,
    mo,
    mode_selector,
    pooling_selector,
    recon_slider,
    run_btn,
    seed_input,
    stage_selector,
    subjs_input,
    wandb_checkbox,
):
    top_controls = [label_selector, mode_selector]
    if mode_selector.value == "Single Fold (CV)":
        top_controls.append(fold_slider)

    control_panel = mo.vstack([
        mo.hstack(top_controls, justify="start"),
        mo.hstack([stage_selector, backbone_lr_dropdown], justify="start"),
        mo.hstack([epochs_slider, lr_dropdown, batch_size_dropdown, subjs_input, seed_input], justify="start"),
        mo.hstack([pooling_selector, dropout_slider, recon_slider], justify="start"),
        mo.hstack([dapt_mask_spatial_slider, dapt_mask_temporal_slider, dapt_mask_context_slider], justify="start"),
        mo.hstack([dapt_weights_input, dapt_ckpt_name_input], justify="start"),
        mo.hstack([dapt_ckpt_input, encoder_lr_dropdown], justify="start"),
        mo.hstack([wandb_checkbox, run_btn], justify="start"),
    ])

    controls_accordion = mo.accordion({"⚙️ Experiment Configuration & Hyperparameters": control_panel})
    controls_accordion
    return


@app.cell
def _(
    SimpleNamespace,
    backbone_lr_dropdown,
    batch_size_dropdown,
    clean_gpu_memory,
    dapt_ckpt_input,
    dapt_ckpt_name_input,
    dapt_mask_context_slider,
    dapt_mask_spatial_slider,
    dapt_mask_temporal_slider,
    dapt_weights_input,
    dropout_slider,
    encoder_lr_dropdown,
    epochs_slider,
    fold_slider,
    label_selector,
    lr_dropdown,
    mo,
    mode_selector,
    pooling_selector,
    recon_slider,
    run_btn,
    run_cross_validation,
    run_dapt,
    run_single_fold,
    run_single_split,
    seed_input,
    stage_selector,
    subjs_input,
    wandb_checkbox,
):
    # Outputs are always defined so downstream cells can read them before
    # training starts. Marimo compiles each cell body on its own, so an early
    # `return` is not allowed here; the work is gated with `if` instead.
    config, history, test_acc, test_f1 = None, None, None, None

    if run_btn.value:
        clean_gpu_memory()

        is_cv_mode = mode_selector.value == "Full Cross-Validation (K-Fold)"
        is_fold_mode = mode_selector.value == "Single Fold (CV)"

        # ── Training stage → mode / freeze / encoder-lr / DAPT checkpoint ────────
        stage = stage_selector.value
        is_dapt = stage == "dapt"
        freeze_perceiver = stage == "stage1"
        encoder_lr = float(encoder_lr_dropdown.value) if stage == "stage3_2" else 0.0
        dapt_ckpt = (dapt_ckpt_input.value.strip() or None) if stage == "stage3_2" else None

        args = SimpleNamespace(
            label=label_selector.value,
            n_epochs=int(epochs_slider.value),
            batch_size=int(batch_size_dropdown.value),
            learning_rate=float(lr_dropdown.value),
            backbone_lr=float(backbone_lr_dropdown.value),
            freeze_perceiver=freeze_perceiver,
            n_subjs=subjs_input.value,
            n_folds=5,
            fold_idx=int(fold_slider.value) if is_fold_mode else None,
            test_frac=0.15,
            val_frac=0.15,
            seed=int(seed_input.value),
            use_wandb=bool(wandb_checkbox.value),
            pooling=str(pooling_selector.value),
            dropout=float(dropout_slider.value),
            recon_weight=float(recon_slider.value),
            cv=is_cv_mode,
            ddp=False,
            # ── Stage 3 (DAPT) ──
            mode="dapt" if is_dapt else "cls",
            dapt_ckpt=dapt_ckpt,
            dapt_mask_ratio=None,
            dapt_mask_ratio_spatial=float(dapt_mask_spatial_slider.value),
            dapt_mask_ratio_temporal=float(dapt_mask_temporal_slider.value),
            dapt_mask_ratio_context=float(dapt_mask_context_slider.value),
            dapt_weights=str(dapt_weights_input.value or "1,1,1,1"),
            dapt_ckpt_name=(dapt_ckpt_name_input.value.strip() or None),
            encoder_lr=encoder_lr,
            # ── Robustness (data validation / DAPT guard) ──
            grad_clip=1.0,
            max_abs=1e4,
            val_mask_seed=1234,
        )

        try:
            if is_dapt:
                spinner_msg = "Running DAPT (Stage 3.1) reconstruction pretraining..."
            elif is_cv_mode:
                spinner_msg = "Running 5-Fold Cross-Validation..."
            else:
                spinner_msg = f"Training Perceiver Classifier ({mode_selector.value})..."

            with mo.status.spinner(title=spinner_msg):
                if is_dapt:
                    # Stage 3.1 is a single reconstruction-pretraining run.
                    _ckpt_path, history = run_dapt(args)
                    test_acc, test_f1 = None, None
                elif is_cv_mode:
                    test_acc, test_f1, history = run_cross_validation(args)
                elif is_fold_mode:
                    test_acc, test_f1, history = run_single_fold(args)
                else:
                    test_acc, test_f1, history = run_single_split(args)
        finally:
            clean_gpu_memory()

        config = vars(args) if hasattr(args, "__dict__") else {}

    return config, history, test_acc, test_f1


@app.cell
def _(os):
    import glob
    import json
    import re
    from datetime import datetime

    import yaml

    from OCD_symphony.utils import proj_dir

    def scan_result_runs():
        """List persisted run-history JSON files under results/runs/."""
        runs = []
        runs_dir = os.path.join(proj_dir, "results", "runs")
        if os.path.isdir(runs_dir):
            for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
                runs.append({
                    "kind": "results",
                    "run_id": os.path.splitext(os.path.basename(path))[0],
                    "path": path,
                    "mtime": os.path.getmtime(path),
                    "config": {},
                    "summary": {},
                })
        return runs

    def scan_wandb_runs():
        """Discover local W&B run directories that contain a console log."""
        roots = [
            os.path.join(proj_dir, "wandb"),
            os.path.join(proj_dir, "notebooks", "wandb"),
        ]
        runs, seen = [], set()
        for root in roots:
            dirs = glob.glob(os.path.join(root, "run-*")) + glob.glob(
                os.path.join(root, "offline-run-*")
            )
            for run_dir in sorted(dirs):
                run_id = os.path.basename(run_dir).split("-")[-1]
                log_path = os.path.join(run_dir, "files", "output.log")
                if run_id in seen or not os.path.isfile(log_path):
                    continue
                seen.add(run_id)

                config, summary = {}, {}
                cfg_path = os.path.join(run_dir, "files", "config.yaml")
                if os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path) as f:
                            raw = yaml.safe_load(f) or {}
                        config = {
                            k: (v["value"] if isinstance(v, dict) and "value" in v else v)
                            for k, v in raw.items() if k != "_wandb"
                        }
                    except Exception:
                        config = {}
                summ_path = os.path.join(run_dir, "files", "wandb-summary.json")
                if os.path.isfile(summ_path):
                    try:
                        with open(summ_path) as f:
                            summary = json.load(f)
                    except Exception:
                        summary = {}

                runs.append({
                    "kind": "wandb",
                    "run_id": run_id,
                    "path": log_path,
                    "mtime": os.path.getmtime(log_path),
                    "config": config,
                    "summary": summary,
                })
        runs.sort(key=lambda r: r["mtime"], reverse=True)
        return runs

    _TRAIN_RE = re.compile(
        r"\[Epoch\s+(\d+)\]\s+Train Loss:\s*([0-9.eE+-]+).*?\|\s*Acc:\s*([0-9.eE+-]+)\s*\|\s*F1:\s*([0-9.eE+-]+)"
    )
    _VAL_RE = re.compile(
        r"\[Epoch\s+(\d+)\]\s+Val Acc:\s*([0-9.eE+-]+)\s*\|\s*Val F1:\s*([0-9.eE+-]+)"
    )
    _TEST_RE = re.compile(r"Test Acc:\s*([0-9.eE+-]+)\s*\|\s*Test F1:\s*([0-9.eE+-]+)")
    _DAPT_RE = re.compile(
        r"\[DAPT Epoch\s+(\d+)\]\s+Train Recon:\s*([0-9.eE+-]+)\s*\|\s*Val Recon:\s*([0-9.eE+-]+)"
    )
    # Optional per-term line printed right after each DAPT epoch line:
    #   ↳ train sp=.. te=.. co=.. pe=.. | val sp=.. te=.. co=.. pe=.. | gnorm=.. | skipped=..
    _NUM = r"([0-9.eE+-]+|nan|inf)"
    _DAPT_TERMS_RE = re.compile(
        r"↳ train sp=" + _NUM + r" te=" + _NUM + r" co=" + _NUM + r" pe=" + _NUM
        + r" \| val sp=" + _NUM + r" te=" + _NUM + r" co=" + _NUM + r" pe=" + _NUM
        + r" \| gnorm=" + _NUM + r" \| skipped=(\d+)"
    )

    def parse_output_log(text):
        """Rebuild a per-epoch history dict from a W&B console log."""
        # Stage 3.1 (DAPT) logs use a different format than classification runs.
        _dapt_matches = list(_DAPT_RE.finditer(text))
        if _dapt_matches:
            _hist = {
                "is_dapt": True,
                "train_recon": [float(m.group(2)) for m in _dapt_matches],
                "val_recon": [float(m.group(3)) for m in _dapt_matches],
            }
            # Per-term breakdown (only present for runs made after the logging update).
            _term_matches = list(_DAPT_TERMS_RE.finditer(text))
            if len(_term_matches) == len(_dapt_matches):
                for _i, _name in enumerate(("spatial", "temporal", "conv", "perceiver")):
                    _hist[f"train_{_name}"] = [float(m.group(1 + _i)) for m in _term_matches]
                    _hist[f"val_{_name}"] = [float(m.group(5 + _i)) for m in _term_matches]
                _hist["grad_norm"] = [float(m.group(9)) for m in _term_matches]
                _hist["skipped_batches"] = [int(m.group(10)) for m in _term_matches]
            return (_hist, None, None)

        train, val = {}, {}
        for match in _TRAIN_RE.finditer(text):
            train[int(match.group(1))] = (
                float(match.group(2)), float(match.group(3)), float(match.group(4)),
            )
        for match in _VAL_RE.finditer(text):
            val[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))

        epochs = sorted(train.keys())
        if not epochs:
            return None, None, None

        history = {
            "train_loss": [train[e][0] for e in epochs],
            "train_acc": [train[e][1] for e in epochs],
            "train_f1": [train[e][2] for e in epochs],
            "val_acc": [val.get(e, (0.0, 0.0))[0] for e in epochs],
            "val_f1": [val.get(e, (0.0, 0.0))[1] for e in epochs],
        }
        test_acc = test_f1 = None
        test_match = _TEST_RE.search(text)
        if test_match:
            test_acc = float(test_match.group(1))
            test_f1 = float(test_match.group(2))
        return history, test_acc, test_f1

    def load_run_json(path):
        """Load a run-history JSON written by finetuning.save_run_history."""
        with open(path) as f:
            payload = json.load(f)
        return (
            payload.get("history"),
            payload.get("test_acc"),
            payload.get("test_f1"),
            payload.get("meta", {}),
        )

    def run_label(meta):
        """Build a human-friendly dropdown label for a run's metadata."""
        cfg = meta.get("config", {}) or {}
        parts = [str(meta.get("run_id", "run"))]
        for key in ("label", "pooling"):
            if cfg.get(key) not in (None, ""):
                parts.append(str(cfg[key]))
        if cfg.get("learning_rate") is not None:
            parts.append(f"lr{float(cfg['learning_rate']):.0e}")
        if cfg.get("n_epochs") is not None:
            parts.append(f"{cfg['n_epochs']}ep")
        if cfg.get("mode") == "dapt":
            parts.append("dapt")
        elif cfg.get("freeze_perceiver") is not None:
            parts.append("stage1" if cfg["freeze_perceiver"] else "stage2")
        if meta.get("mtime"):
            parts.append(datetime.fromtimestamp(meta["mtime"]).strftime("%Y-%m-%d %H:%M"))
        return " · ".join(parts)

    return (
        load_run_json,
        parse_output_log,
        run_label,
        scan_result_runs,
        scan_wandb_runs,
    )


@app.cell
def _():
    _HYPERPARAM_ORDER = [
        "label", "mode", "n_epochs", "batch_size", "learning_rate", "backbone_lr",
        "encoder_lr", "freeze_perceiver", "pooling", "dropout", "recon_weight",
        "dapt_mask_ratio", "dapt_mask_ratio_spatial", "dapt_mask_ratio_temporal",
        "dapt_mask_ratio_context", "dapt_weights", "dapt_ckpt", "dapt_ckpt_name",
        "grad_clip", "max_abs", "val_mask_seed",
        "n_subjs", "n_folds", "fold_idx", "test_frac", "val_frac",
        "seed", "cv", "ddp", "use_wandb",
    ]
    _HYPERPARAM_DEFS = {
        "label": "Target variable for classification (e.g. gender, age).",
        "mode": "'cls' = supervised classification; 'dapt' = Stage 3.1 reconstruction pretraining.",
        "n_epochs": "Number of full passes over the training set.",
        "batch_size": "Number of samples per gradient update.",
        "learning_rate": "Learning rate for the classification head (also DAPT LR in Stage 3.1).",
        "backbone_lr": "Learning rate for the Perceiver backbone (Stage 2 only).",
        "encoder_lr": "Learning rate for spatial/temporal/conv encoders (Stage 3.2); 0 = frozen.",
        "freeze_perceiver": "True freezes the Perceiver (linear probe); False fine-tunes it.",
        "pooling": "Latent pooling strategy: mean / attention / flatten.",
        "dropout": "Dropout rate in the classification head.",
        "recon_weight": "Reconstruction loss balance α: (1−α)·cls + α·recon.",
        "dapt_mask_ratio": "Shared DAPT masking ratio (overrides the per-encoder values when set).",
        "dapt_mask_ratio_spatial": "DAPT masking ratio for the spatial encoder (Stage 3.1).",
        "dapt_mask_ratio_temporal": "DAPT masking ratio for the temporal encoder (Stage 3.1).",
        "dapt_mask_ratio_context": "DAPT masking ratio for the context/Conv1D encoder (Stage 3.1).",
        "dapt_weights": "DAPT reconstruction-loss weights: spatial,temporal,conv,perceiver.",
        "dapt_ckpt": "Stage 3.1 checkpoint loaded as the Stage 3.2 initialisation.",
        "dapt_ckpt_name": "Optional filename override for the saved Stage 3.1 checkpoint.",
        "grad_clip": "Global gradient-norm clipping threshold in DAPT (0 = disabled).",
        "max_abs": "Data validation: runs with |fMRI| above this (or non-finite) are dropped.",
        "val_mask_seed": "Seed of the deterministic DAPT validation masks.",
        "n_subjs": "Number of subjects loaded (None = all available).",
        "n_folds": "Number of leave-subject-out cross-validation folds.",
        "fold_idx": "Specific CV fold index (0-based); None = all folds.",
        "test_frac": "Fraction of subjects held out for testing.",
        "val_frac": "Fraction of subjects held out for validation.",
        "seed": "Random seed for reproducibility.",
        "cv": "Whether full K-fold cross-validation was used.",
        "ddp": "Whether DistributedDataParallel multi-GPU training was used.",
        "use_wandb": "Whether the run was logged to Weights & Biases.",
    }

    def _format_hyperparam_value(value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, float):
            return f"{value:.0e}" if 0 < abs(value) < 1e-2 else f"{value:g}"
        return str(value)

    def hyperparameters_table(config, mo):
        """Render a (name, value, definition) table for a run's config."""
        if not config:
            return None
        keys = [k for k in _HYPERPARAM_ORDER if k in config]
        keys += [
            k for k in config
            if k not in _HYPERPARAM_ORDER and not str(k).startswith("_")
        ]
        if not keys:
            return None
        rows = ["| Hyperparameter | Value | Definition |", "| :--- | :---: | :--- |"]
        for key in keys:
            value = _format_hyperparam_value(config[key])
            definition = _HYPERPARAM_DEFS.get(key, "Run configuration parameter.")
            rows.append(f"| `{key}` | {value} | {definition} |")
        return mo.md("#### ⚙️ Hyperparameters\n" + "\n".join(rows))

    def render_history(history, test_acc, test_f1, mo, plt, config=None):
        """Render a metrics dashboard (DAPT recon, CV summary, or single-run curves)."""
        is_dapt = isinstance(history, dict) and history.get("is_dapt", False)
        is_cv = isinstance(history, dict) and history.get("is_cv", False)

        _hyperparam_section = hyperparameters_table(config, mo)
        prelude = [_hyperparam_section] if _hyperparam_section is not None else []

        if is_dapt:
            train_recon = history.get("train_recon", [])
            val_recon = history.get("val_recon", [])
            best_val = min(val_recon) if val_recon else None
            best_epoch = (val_recon.index(best_val) + 1) if best_val is not None else None

            metric_cards = mo.hstack([
                mo.stat(
                    label="Final Train Recon",
                    value=f"{train_recon[-1]:.4f}" if train_recon else "N/A",
                    caption="Weighted masked-reconstruction loss",
                ),
                mo.stat(
                    label="Best Val Recon",
                    value=f"{best_val:.4f}" if best_val is not None else "N/A",
                    caption=f"Epoch {best_epoch}" if best_epoch else "—",
                ),
                mo.stat(
                    label="Epochs Completed",
                    value=f"{len(train_recon)}",
                    caption="DAPT (Stage 3.1)",
                ),
            ], justify="space-around")

            epochs = list(range(1, len(train_recon) + 1))
            _terms = ("spatial", "temporal", "conv", "perceiver")
            _term_colors = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")
            _has_terms = all(
                len(history.get(f"train_{t}", [])) == len(train_recon) and len(train_recon) > 0
                for t in _terms
            )

            if _has_terms:
                # 2x2: total | per-term train | per-term val | grad-norm & skipped
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                ax = axes[0, 0]
                ax.plot(epochs, train_recon, marker="o", color="#1f77b4", label="Train Recon")
                ax.plot(epochs, val_recon, marker="s", color="#ff7f0e", label="Val Recon")
                ax.set_title("Total weighted reconstruction loss", fontsize=11, fontweight="bold")
                ax.set_ylabel("Weighted MSE")

                for _ax, _split in ((axes[0, 1], "train"), (axes[1, 0], "val")):
                    for _t, _c in zip(_terms, _term_colors):
                        _ax.plot(epochs, history[f"{_split}_{_t}"], marker="o", color=_c, label=_t)
                    _ax.set_title(f"Per-term {_split} loss", fontsize=11, fontweight="bold")
                    _ax.set_ylabel("MSE")
                    _ax.set_yscale("log")

                ax = axes[1, 1]
                _gn = history.get("grad_norm", [])
                if _gn:
                    ax.plot(epochs, _gn, marker="o", color="#8c564b", label="grad norm")
                    ax.set_ylabel("Mean grad norm")
                _sk = history.get("skipped_batches", [])
                if _sk and any(_sk):
                    ax2 = ax.twinx()
                    ax2.bar(epochs, _sk, alpha=0.3, color="#e377c2", label="skipped batches")
                    ax2.set_ylabel("Skipped batches")
                ax.set_title("Gradient norm / skipped batches", fontsize=11, fontweight="bold")

                for _ax in axes.flat:
                    _ax.set_xlabel("Epoch")
                    _ax.grid(True, linestyle="--", alpha=0.6)
                    if _ax.get_legend_handles_labels()[0]:
                        _ax.legend(fontsize=8)
            else:
                # Older runs (no per-term history): single total-loss plot.
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                ax.plot(epochs, train_recon, marker="o", color="#1f77b4", label="Train Recon")
                ax.plot(epochs, val_recon, marker="s", color="#ff7f0e", label="Val Recon")
                ax.set_title("DAPT Reconstruction Loss", fontsize=12, fontweight="bold")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Weighted MSE")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()
            plt.tight_layout()

            _ckpt_note = history.get("ckpt_path")
            _extra = [mo.md(f"💾 **Checkpoint:** `{_ckpt_note}`")] if _ckpt_note else []

            # ── Data-validation exclusions (bad subjects / runs) ─────────────
            _excl = history.get("excluded_data") or []
            if _excl:
                _n_subj = sum(1 for e in _excl if e.get("run") is None)
                _n_runs = len(_excl) - _n_subj
                _rows = [
                    "| Subject | Run | Reason |",
                    "| :--- | :---: | :--- |",
                ] + [
                    f"| `{e.get('subject')}` | "
                    f"{'ALL' if e.get('run') is None else e.get('run')} | {e.get('reason')} |"
                    for e in _excl
                ]
                _extra.append(
                    mo.accordion({
                        f"⚠️ Data validation: {_n_subj} subject(s) excluded, "
                        f"{_n_runs} run(s) dropped": mo.md("\n".join(_rows))
                    })
                )
            _skipped_total = sum(history.get("skipped_batches", []) or [])
            if _skipped_total:
                _extra.append(
                    mo.md(f"⚠️ **{_skipped_total}** training batch(es) were skipped by the "
                          "non-finite loss/gradient guard.")
                )

            return mo.vstack(prelude + [
                mo.md("### 🧬 DAPT (Stage 3.1) Reconstruction Results"),
                metric_cards,
                mo.as_html(fig),
            ] + _extra)

        if is_cv:
            mean_acc = history["mean_acc"]
            std_acc = history["std_acc"]
            mean_f1 = history["mean_f1"]
            std_f1 = history["std_f1"]
            fold_accs = history["fold_accs"]
            fold_f1s = history["fold_f1s"]
            histories = history.get("histories", [])

            metric_cards = mo.hstack([
                mo.stat(
                    label="Mean Test Accuracy",
                    value=f"{mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%",
                    caption=f"Across {len(fold_accs)} CV Folds",
                ),
                mo.stat(
                    label="Mean Test Macro F1",
                    value=f"{mean_f1:.4f} ± {std_f1:.4f}",
                    caption="Leave-Subject-Out generalization",
                ),
                mo.stat(
                    label="Completed Folds",
                    value=f"{len(fold_accs)} / 5",
                    caption="Total cross-validation splits",
                ),
            ], justify="space-around")

            table_rows = []
            for idx, (f_acc, f_f1) in enumerate(zip(fold_accs, fold_f1s)):
                table_rows.append(f"| Fold {idx + 1} | {f_acc * 100:.2f}% | {f_f1:.4f} |")
            table_rows.append(
                f"| **Mean ± Std** | **{mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%** | **{mean_f1:.4f} ± {std_f1:.4f}** |"
            )

            table_md = mo.md(
                "#### 📋 Per-Fold Results\n"
                "| Split | Test Accuracy | Test Macro F1 |\n"
                "| :--- | :---: | :---: |\n" + "\n".join(table_rows)
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            for f_idx, fold_hist in enumerate(histories):
                if "train_loss" in fold_hist and len(fold_hist["train_loss"]) > 0:
                    ep = list(range(1, len(fold_hist["train_loss"]) + 1))
                    ax1.plot(ep, fold_hist["train_loss"], alpha=0.7, label=f"Fold {f_idx + 1}")
                    ax2.plot(ep, [a * 100 for a in fold_hist["val_acc"]], alpha=0.7, label=f"Fold {f_idx + 1}")

            ax1.set_title("Training Loss Across Folds", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()

            ax2.set_title("Validation Accuracy Dynamics", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Val Acc (%)")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()
            plt.tight_layout()

            dashboard_view = mo.vstack(prelude + [
                mo.md("### 📊 5-Fold Cross-Validation Summary"),
                metric_cards,
                table_md,
                mo.as_html(fig),
            ])
        else:
            acc_str = f"{test_acc * 100:.2f}%" if test_acc is not None else "N/A"
            f1_str = f"{test_f1:.4f}" if test_f1 is not None else "N/A"
            metric_cards = mo.hstack([
                mo.stat(
                    label="Test Accuracy",
                    value=acc_str,
                    caption="Held-out subject test set",
                ),
                mo.stat(
                    label="Test Macro F1",
                    value=f1_str,
                    caption="Balanced classification performance",
                ),
                mo.stat(
                    label="Epochs Completed",
                    value=f"{len(history['train_loss'])}",
                    caption="Full iterations",
                ),
            ], justify="space-around")

            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

            ax1.plot(epochs, history["train_loss"], marker="o", color="#1f77b4", label="Train Loss")
            ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss (CrossEntropy + MSE)")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()

            ax2.plot(epochs, [a * 100 for a in history["train_acc"]], marker="o", color="#2ca02c", label="Train Acc (%)")
            ax2.plot(epochs, [a * 100 for a in history["val_acc"]], marker="s", color="#ff7f0e", label="Val Acc (%)")
            ax2.plot(epochs, [f * 100 for f in history["val_f1"]], marker="^", color="#9467bd", linestyle=":", label="Val F1 (%)")
            ax2.set_title("Accuracy & F1 Dynamics", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score (%)")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()

            plt.tight_layout()

            dashboard_view = mo.vstack(prelude + [
                mo.md("### 📊 Experiment Results"),
                metric_cards,
                mo.as_html(fig),
            ])

        return dashboard_view

    return (render_history,)


@app.cell
def _(config, history, mo, plt, render_history, test_acc, test_f1):
    _is_dapt = isinstance(history, dict) and history.get("is_dapt", False)
    mo.stop(
        test_acc is None and not _is_dapt,
        mo.md("👉 *Configure parameters above and click **Start Fine-tuning** to run.*"),
    )

    live_view = render_history(history, test_acc, test_f1, mo, plt, config=config)
    live_view
    return


@app.cell(hide_code=True)
def _(mo):
    # ── DAPT reconstruction controls ─────────────────────────────────────────
    recon_roi_strategy = mo.ui.dropdown(
        options={
            "One ROI per network": "network",
            "Best R² ROIs": "best",
            "Worst R² ROIs": "worst",
            "Median R² ROIs": "median",
        },
        value="One ROI per network",
        label="ROI selection",
    )
    recon_n_rois = mo.ui.slider(start=2, stop=20, value=10, step=1, label="Number of ROIs")
    recon_fraction = mo.ui.slider(
        start=10, stop=100, step=10, value=10,
        label="Fraction of held-out test subjects (%)",
    )
    recon_plot_width = mo.ui.slider(
        start=400, stop=1200, step=50, value=600, label="Plot width (px)",
    )
    # Index of the sample within the selected subject (clamped in the viewer cell).
    recon_sample_idx = mo.ui.number(start=0, stop=5000, value=0, step=1, label="Sample index in subject")
    # Counter-valued button: every click increments .value. Changing other controls
    # re-runs dependent cells but does not change the counter, so no inference starts.
    recon_run_btn = mo.ui.button(
        value=0,
        on_click=lambda clicks: clicks + 1,
        label="🧠 Reconstruct BOLD (run inference)",
        kind="success",
    )
    return (
        recon_fraction,
        recon_n_rois,
        recon_plot_width,
        recon_roi_strategy,
        recon_run_btn,
        recon_sample_idx,
    )


@app.cell
def _(mo):
    # Latest reconstruction only, keyed by (checkpoint, device, fraction), so changing
    # plot controls does not re-run the model. Lives for the marimo session.
    get_recon_cache, set_recon_cache = mo.state({})
    # Number of Reconstruct clicks already processed (see recon_run_btn).
    get_recon_clicks, set_recon_clicks = mo.state(0)
    return get_recon_cache, get_recon_clicks, set_recon_cache, set_recon_clicks


@app.cell
def _(mo):
    # Loaded-run checkpoint, created once in its own cell so it is not reset when
    # the run selector re-runs.
    get_loaded_ckpt, set_loaded_ckpt = mo.state(None)
    return get_loaded_ckpt, set_loaded_ckpt


@app.cell
def _(load_btn, run_metas, run_selector, set_loaded_ckpt):
    # Store the checkpoint of the loaded DAPT run in state when Load is clicked.
    # Handles both run kinds: "results" (JSON history carries ckpt_path) and
    # "wandb" (parsed from output.log, which has no ckpt_path, so recover it from
    # the "[DAPT] Saved checkpoint to ..." line or the matching results JSON).
    import glob as _glob
    import json as _json
    import os as _os
    import re as _re
    from OCD_symphony.utils import proj_dir as _proj_dir

    def _find_ckpt(_meta):
        _cands = []
        if _meta.get("kind") == "results":
            with open(_meta["path"]) as _f:
                _cands.append((_json.load(_f).get("history") or {}).get("ckpt_path"))
        else:
            with open(_meta["path"], errors="replace") as _f:
                _m = _re.findall(r"\[DAPT\] Saved checkpoint to (\S+)", _f.read())
            if _m:
                _cands.append(_m[-1])
            # Fallback: results JSON saved with the same W&B id suffix.
            for _p in _glob.glob(_os.path.join(_proj_dir, "results", "runs", f"*_{_meta['run_id']}.json")):
                with open(_p) as _f:
                    _cands.append((_json.load(_f).get("history") or {}).get("ckpt_path"))
            # Fallback: explicit checkpoint name from the run config.
            _name = (_meta.get("config") or {}).get("dapt_ckpt_name")
            if _name:
                _cands.append(_os.path.join(_proj_dir, "models", _name))
        return next((_c for _c in _cands if _c and _os.path.isfile(_c)), None)

    if run_selector is not None and load_btn is not None and load_btn.value:
        _meta = run_metas.get(run_selector.value)
        if _meta is not None:
            _ck = _find_ckpt(_meta)
            if _ck:
                set_loaded_ckpt(_ck)



@app.cell
def _(plt):
    import io as _io

    def fig_to_png(fig, dpi=120):
        """Rasterise a matplotlib figure to PNG bytes (and close it) so it can be
        shown with a controlled pixel width via ``mo.image``."""
        _buf = _io.BytesIO()
        fig.savefig(_buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return _buf.getvalue()

    return (fig_to_png,)


@app.cell
def _(
    get_loaded_ckpt,
    get_recon_cache,
    get_recon_clicks,
    mo,
    recon_fraction,
    recon_run_btn,
    set_recon_cache,
    set_recon_clicks,
):
    # ── Reconstruction: inference runs only when the button is clicked ───────
    from OCD_symphony.reconstruction.dapt_reconstruction import (
        default_device, estimate_seconds_per_subject, fit_normalisation_stats,
        reconstruct_subject_stats, select_test_subject_subset,
    )

    RECON_KEEP_K = 6  # best + worst + (K-2) median-scoring subjects kept in memory

    selected_ckpt = get_loaded_ckpt()
    _device = default_device()
    _title = mo.md("### 🧠 DAPT Reconstruction")
    recon_result = None

    if selected_ckpt is None:
        recon_header = mo.vstack([
            _title,
            mo.md(
                "*Select a DAPT run with a saved checkpoint and click **Load & Visualize History** "
                "to inspect its BOLD reconstructions.*"
            ),
        ])
    else:
        _frac = recon_fraction.value / 100.0
        _stats = fit_normalisation_stats(n_subjs=500)
        _n_test_total = len(_stats["test_subjs"])
        _n_subj_req = max(1, round(_frac * _n_test_total))
        _eta_s = (_n_subj_req + RECON_KEEP_K) * estimate_seconds_per_subject(_device)
        _eta_txt = f"~{_eta_s:.0f} s" if _eta_s < 120 else f"~{_eta_s / 60:.1f} min"
        _key = (selected_ckpt, str(_device), recon_fraction.value)
        _cache = get_recon_cache()

        _clicks = recon_run_btn.value
        if _clicks > get_recon_clicks():
            set_recon_clicks(_clicks)
            _subjs = select_test_subject_subset(_stats["test_subjs"], frac=_frac, seed=0)
            _total = len(_subjs) + RECON_KEEP_K
            with mo.status.progress_bar(
                total=_total,
                title="Reconstructing BOLD",
                subtitle=f"{len(_subjs)} subjects on {_device.type.upper()} (est. {_eta_txt})",
                completion_title="Reconstruction complete",
                completion_subtitle=f"{len(_subjs)} subjects",
            ) as _bar:
                _res = reconstruct_subject_stats(
                    selected_ckpt, _subjs, _stats["W"], _stats["B"],
                    keep_k=RECON_KEEP_K, device=_device,
                    progress=lambda done, total: _bar.update(),
                )
            _res["fraction"] = recon_fraction.value
            # Keep only the latest reconstruction in memory.
            _cache = {_key: _res}
            set_recon_cache(_cache)

        recon_result = _cache.get(_key)
        if recon_result is None:
            _status = mo.md(
                f"👉 *Click **Reconstruct BOLD** to run the model on {recon_fraction.value}% of the "
                f"held-out test subjects ({_n_subj_req} subjects, est. {_eta_txt}).*"
            )
        else:
            _status = mo.md(
                f"✅ Reconstruction ready: **{recon_result['subjs'].shape[0]}** samples from "
                f"**{len(set(recon_result['subjs'].tolist()))}** subjects "
                f"({recon_fraction.value}% of {_n_test_total} test subjects). "
                f"Traces kept in memory for {len(recon_result['representative'])} representative subjects."
            )
        recon_header = mo.vstack([
            _title,
            mo.hstack([recon_fraction, recon_run_btn], justify="start", wrap=True),
            mo.md(
                f"Device: **{_device.type.upper()}** · selection: **{recon_fraction.value}%** = "
                f"**{_n_subj_req}** of {_n_test_total} test subjects · estimated time **{_eta_txt}**"
            ),
            _status,
        ])
    recon_header
    return (recon_result,)


@app.cell
def _(fig_to_png, mo, recon_plot_width, recon_result):
    # ── Overall reconstruction statistics (all reconstructed samples) ────────
    mo.stop(recon_result is None)

    import numpy as _np
    from OCD_symphony.data_validation.validate_preprocessing import plot_recon_summary

    recon_r2 = recon_result["r2"]
    _corr = recon_result["corr"]
    _subjs = recon_result["subjs"]
    if "summary_png" not in recon_result:
        recon_result["summary_png"] = fig_to_png(plot_recon_summary(recon_r2, _corr, _subjs))

    _n_samples = _corr.shape[0]
    _n_subj = len(set(_subjs.tolist()))
    _cards = mo.hstack([
        mo.stat(label="Mean ROI R²", value=f"{recon_r2.mean():.3f}", caption=f"Median {_np.median(recon_r2):.3f}"),
        mo.stat(
            label="ROIs with R² > 0",
            value=f"{100 * (recon_r2 > 0).mean():.1f}%",
            caption=f"{int((recon_r2 > 0).sum())} / {recon_r2.size} ROIs",
        ),
        mo.stat(
            label="Mean Pearson R",
            value=f"{_np.nanmean(_corr):.3f}",
            caption="Over samples × ROIs (time-wise)",
        ),
        mo.stat(label="Samples", value=f"{_n_samples}", caption=f"{_n_subj} subjects"),
    ], justify="space-around")

    summary_view = mo.vstack([
        mo.md("#### 📊 Overall reconstruction statistics"),
        _cards,
        mo.image(recon_result["summary_png"], width=recon_plot_width.value),
    ])
    summary_view
    return (recon_r2,)


@app.cell
def _(mo, recon_result):
    # ── Subject picker: representative subjects kept in memory ──────────────
    mo.stop(recon_result is None)

    _opts = {}
    _reps = recon_result["representative"]
    for _pos, _sid in enumerate(_reps):
        if _pos == 0:
            _role = "best"
        elif _pos == len(_reps) - 1:
            _role = "worst"
        else:
            _role = "median"
        _opts[f"{_sid} · {_role}"] = _sid
    recon_subject = mo.ui.dropdown(
        options=_opts, value=next(iter(_opts)), label="Subject (representative)",
    )
    return (recon_subject,)


@app.cell
def _(
    fig_to_png,
    mo,
    recon_n_rois,
    recon_plot_width,
    recon_r2,
    recon_result,
    recon_roi_strategy,
    recon_sample_idx,
    recon_subject,
):
    # ── Single-sample BOLD trace viewer (first 200 TRs) ──────────────────────
    mo.stop(recon_result is None)

    from OCD_symphony.data_validation.validate_preprocessing import (
        plot_recon_timeseries, select_rois_for_display,
    )

    _sid = recon_subject.value
    _tr = recon_result["traces"][_sid]
    _n = _tr["orig"].shape[0]
    _T = _tr["orig"].shape[2]
    _i = min(max(int(recon_sample_idx.value), 0), _n - 1)
    _rois = select_rois_for_display(recon_r2, n_rois=recon_n_rois.value, strategy=recon_roi_strategy.value)
    _png = fig_to_png(
        plot_recon_timeseries(
            _tr["orig"], _tr["recon"], sample_idx=_i,
            roi_indices=_rois, t_window=(0, min(200, _T)),
        )
    )
    trace_view = mo.vstack([
        mo.md("#### 🧬 BOLD traces (single sample)"),
        mo.hstack([recon_roi_strategy, recon_n_rois], justify="start", wrap=True),
        mo.hstack([recon_subject, recon_sample_idx, recon_plot_width], justify="start", wrap=True),
        mo.md(
            f"Subject `{_sid}` (sample {_i + 1}/{_n}) · "
            f"mean ROI-wise R² over the reconstructed set: **{recon_r2.mean():.3f}** · "
            f"panel titles show per-trace Pearson r · TRs 0-{min(200, _T)}."
        ),
        mo.image(_png, width=recon_plot_width.value),
    ])
    trace_view
    return


@app.cell
def _(mo, run_label, scan_result_runs, scan_wandb_runs):
    # ── Discover previous runs from disk (results/runs/ JSON + local W&B dirs) ──
    run_metas = {}
    for _run in scan_result_runs() + scan_wandb_runs():
        _meta = {
            "kind": _run["kind"],
            "run_id": _run["run_id"],
            "path": _run["path"],
            "mtime": _run["mtime"],
            "config": _run.get("config", {}),
            "summary": _run.get("summary", {}),
        }
        run_metas[run_label(_meta)] = _meta

    if run_metas:
        _ordered = sorted(run_metas.keys(), reverse=True)
        run_selector = mo.ui.dropdown(
            options={_label: _label for _label in _ordered},
            value=_ordered[0],
            label="Previous Run",
        )
        load_btn = mo.ui.run_button(label="📈 Load & Visualize History")
    else:
        run_selector = None
        load_btn = None

    return load_btn, run_metas, run_selector


@app.cell
def _(
    load_btn,
    load_run_json,
    mo,
    parse_output_log,
    plt,
    render_history,
    run_metas,
    run_selector,
):
    if not run_metas:
        loaded_view = mo.md(
            "⚠️ *No previous runs found. Run a fine-tuning job with W&B logging enabled, "
            "or ensure `results/runs/` contains saved histories.*"
        )
    else:
        _header = mo.hstack([run_selector, load_btn], justify="start")

        if load_btn.value:
            _meta = run_metas[run_selector.value]
            if _meta["kind"] == "results":
                _history, _test_acc, _test_f1, _json_meta = load_run_json(_meta["path"])
                _config = _json_meta.get("config", {})
            else:
                with open(_meta["path"]) as _log_file:
                    _history, _test_acc, _test_f1 = parse_output_log(_log_file.read())
                if _test_acc is None:
                    _test_acc = _meta["summary"].get("test_acc")
                    _test_f1 = _meta["summary"].get("test_f1")
                _config = _meta.get("config", {})

            if _history is None:
                _body = mo.md("⚠️ *Could not parse a history from the selected run.*")
            else:
                _body = mo.vstack([
                    mo.md(f"### 🗂️ Loaded Run: `{_meta['run_id']}`"),
                    render_history(_history, _test_acc, _test_f1, mo, plt, config=_config),
                ])
        else:
            _body = mo.md("👉 *Select a previous run and click **Load & Visualize History**.*")

        loaded_view = mo.vstack([_header, _body])

    loaded_view
    return


if __name__ == "__main__":
    app.run()
