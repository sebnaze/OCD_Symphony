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

    # Force a reload so edits to the fine-tuning package are picked up without restarting
    # the marimo kernel (Python caches imported modules in sys.modules). Dependencies come
    # first: a module's ``from X import Y`` bindings only refresh when it is reloaded after X.
    import OCD_symphony.model.constants as _const
    import OCD_symphony.model.config as _cfg
    import OCD_symphony.model.metrics as _metrics
    import OCD_symphony.model.checkpoints as _ckpt
    import OCD_symphony.model.dapt as _dapt
    import OCD_symphony.model.finetuning as _ft
    import OCD_symphony.model.experiment as _exp
    for _m in (_const, _cfg, _metrics, _ckpt, _dapt, _ft, _exp):
        importlib.reload(_m)

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
    # ── Block 1: training stage and training dataset ──────────────────────────
    stage_selector = mo.ui.dropdown(
        options={
            "Stage 0: Logistic Regression Probe": "probe",
            "Stage 1: Linear Probing": "stage1",
            "Stage 2: Finetuning": "stage2",
            "Stage 3.1: Re-training (DAPT)": "dapt",
            "Stage 3.2: Post-training": "stage3_2",
        },
        value="Stage 2: Finetuning",
        label="Training Stage",
    )
    dataset_selector = mo.ui.dropdown(
        options={"HCP-YA": "hcp_ya", "OCD": "ocd"},
        value="HCP-YA",
        label="Training Dataset",
    )
    mo.hstack([stage_selector, dataset_selector], justify="start", wrap=True)
    return dataset_selector, stage_selector


@app.cell
def _():
    import glob as _glob
    import json as _json
    import os as _os
    import re as _re
    from OCD_symphony.utils import proj_dir as _proj_dir

    def find_ckpt(_meta):
        """Return an existing checkpoint path for a previous run, or None.
        Handles both run kinds: "results" (JSON history carries ckpt_path) and
        "wandb" (parsed from output.log, which has no ckpt_path, so recover it from
        the "[DAPT] Saved checkpoint to ..." line or the matching results JSON)."""
        _cands = []

        def _safe_history_ckpt(_p):
            # One unreadable or malformed run JSON must not break the controls cell.
            try:
                with open(_p) as _f:
                    return (_json.load(_f).get("history") or {}).get("ckpt_path")
            except (OSError, ValueError, AttributeError) as _exc:
                print(f"[find_ckpt] skipped {_p}: {_exc}")
                return None

        if _meta.get("kind") == "results":
            _cands.append(_safe_history_ckpt(_meta["path"]))
        else:
            try:
                with open(_meta["path"], errors="replace") as _f:
                    _m = _re.findall(r"\[DAPT\] Saved checkpoint to (\S+)", _f.read())
                if _m:
                    _cands.append(_m[-1])
            except OSError as _exc:
                print(f"[find_ckpt] skipped {_meta.get('path')}: {_exc}")
            # Fallback: results JSON saved with the same W&B id suffix.
            for _p in _glob.glob(_os.path.join(_proj_dir, "results", "runs", f"*_{_meta['run_id']}.json")):
                _cands.append(_safe_history_ckpt(_p))
            # Fallback: explicit checkpoint name from the run config.
            _name = (_meta.get("config") or {}).get("dapt_ckpt_name")
            if _name:
                _cands.append(_os.path.join(_proj_dir, "models", _name))
        return next((_c for _c in _cands if _c and _os.path.isfile(_c)), None)

    return (find_ckpt,)


@app.cell
def _(dataset_selector, find_ckpt, mo, run_metas, stage_selector):
    # ── Block 2 widgets: rebuilt on stage / dataset change so defaults follow them ──
    _stage_defaults = {
        "probe": dict(epochs=10, lr=1e-4, perceiver_lr=1e-6, encoder_lr=0.0),
        "stage1": dict(epochs=10, lr=1e-4, perceiver_lr=1e-5, encoder_lr=0.0),
        "stage2": dict(epochs=10, lr=1e-4, perceiver_lr=1e-5, encoder_lr=0.0),
        "dapt": dict(epochs=50, lr=1e-4, perceiver_lr=1e-5, encoder_lr=0.0),
        "stage3_2": dict(epochs=10, lr=1e-4, perceiver_lr=1e-6, encoder_lr=0.0),
    }
    _d = _stage_defaults[stage_selector.value]
    # Target choices per dataset (display label -> value). OCD adds the condition target.
    _label_opts = {
        "hcp_ya": {"Gender": "gender", "Age (regression)": "age"},
        "ocd": {"Condition (OCD vs HC)": "condition", "Gender": "gender", "Age (regression)": "age"},
    }
    _label_default = {"hcp_ya": "Gender", "ocd": "Condition (OCD vs HC)"}

    # Previous Run options: Stage 3.1 baseline = finetuned classifier runs;
    # Stage 3.2 = DAPT runs. Only runs with a resolvable checkpoint are listed.
    _baseline_label = "Baseline (pre-trained on HCP-Aging)"
    _cls_opts = {_baseline_label: None}
    _dapt_opts = {}
    for _lbl in run_metas.keys():  # already newest first
        _meta = run_metas[_lbl]
        try:
            _ck = find_ckpt(_meta)
        except Exception as _exc:
            # A broken previous run must not block every widget below Training Stage.
            print(f"[controls] skipped run {_meta.get('path')}: {_exc}")
            continue
        if not _ck:
            continue
        _is_dapt_run = (_meta.get("config") or {}).get("mode") == "dapt"
        (_dapt_opts if _is_dapt_run else _cls_opts)[_lbl] = _ck

    _no_dapt = "No DAPT run with a checkpoint found"
    dapt_init_selector = mo.ui.dropdown(
        options=_cls_opts,
        value=_baseline_label,
        label="Pre-training run",
    )
    post_ckpt_selector = mo.ui.dropdown(
        options=_dapt_opts or {_no_dapt: None},
        value=next(iter(_dapt_opts), _no_dapt),
        label="Pre-training run",
    )
    # Stage 0 encoder init: HCP-AGING baseline (None) or any DAPT run.
    probe_init_selector = mo.ui.dropdown(
        options={"HCP-AGING (baseline)": None, **_dapt_opts},
        value="HCP-AGING (baseline)",
        label="Encoder init",
    )

    label_selector = mo.ui.dropdown(
        options=_label_opts[dataset_selector.value],
        value=_label_default[dataset_selector.value],
        label="Target Label",
    )
    # Stage 1/2 starting weights: HCP-Aging baseline (None) or any DAPT run.
    init_selector = mo.ui.dropdown(
        options={"HCP-AGING (baseline)": None, **_dapt_opts},
        value="HCP-AGING (baseline)",
        label="Starting weights",
    )
    mode_selector = mo.ui.radio(
        options=["Single Split", "Single Fold (CV)", "Full Cross-Validation (K-Fold)"],
        value="Single Split",
        label="Evaluation Mode",
    )
    # Stage 0 only: logistic regression probe evaluation (split or subject-level K-fold CV).
    probe_eval_radio = mo.ui.radio(
        options={"Single Split": "split", "Full Cross-Validation (K-Fold)": "cv"},
        value="Single Split",
        label="Evaluation Mode",
    )
    # Stage 0 only: how Perceiver latents are reduced before the logistic regression.
    probe_pooling_radio = mo.ui.radio(
        options={
            "PCA (256 components)": "pca",
            "Per feature (mean over ROIs)": "feature",
            "Per ROI (mean over features)": "roi",
            "Per network (mean over ROIs per network)": "network",
        },
        value="PCA (256 components)",
        label="Pooling",
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
        value=_d["epochs"],
        step=1,
        label="Epochs",
    )
    # Stage 3.1 uses one optimizer LR for encoders + Perceiver; other stages use the classifier-head LR.
    lr_dropdown = mo.ui.dropdown(
        options=[1e-5, 5e-5, 1e-4, 3e-4, 1e-3],
        value=_d["lr"],
        label="Learning Rate (all modules)" if stage_selector.value == "dapt" else "Classifier LR",
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
    backbone_lr_dropdown = mo.ui.dropdown(
        options=[1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4],
        value=_d["perceiver_lr"],
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
    # ── Stage 3.1 (DAPT) controls ───────────────────────────────────────────
    dapt_mask_spatial_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.7,
        step=0.05,
        label="Spatial",
    )
    dapt_mask_temporal_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.7,
        step=0.05,
        label="Temporal",
    )
    dapt_mask_context_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        value=0.5,
        step=0.05,
        label="Context",
    )
    dapt_weights_input = mo.ui.text(
        value="1,1,1,1",
        label="Loss Weights sp,tp,conv,perceiver",
    )
    dapt_ckpt_name_input = mo.ui.text(
        value="",
        label="Output Name (optional)",
    )
    # ── Stage 3.2 controls ──────────────────────────────────────────────────
    encoder_lr_dropdown = mo.ui.dropdown(
        options=[0.0, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5],
        value=_d["encoder_lr"],
        label="Encoder LR (0 = frozen)",
    )
    run_btn = mo.ui.run_button(
        label="🚀 Start Fine-tuning",
        kind="success",
    )
    return (
        backbone_lr_dropdown,
        batch_size_dropdown,
        dapt_ckpt_name_input,
        dapt_init_selector,
        dapt_mask_context_slider,
        dapt_mask_spatial_slider,
        dapt_mask_temporal_slider,
        dapt_weights_input,
        dropout_slider,
        encoder_lr_dropdown,
        epochs_slider,
        fold_slider,
        init_selector,
        label_selector,
        lr_dropdown,
        mode_selector,
        post_ckpt_selector,
        pooling_selector,
        probe_eval_radio,
        probe_init_selector,
        probe_pooling_radio,
        recon_slider,
        run_btn,
        seed_input,
        subjs_input,
        wandb_checkbox,
    )


@app.cell
def _(
    backbone_lr_dropdown,
    batch_size_dropdown,
    dapt_ckpt_name_input,
    dapt_init_selector,
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
    post_ckpt_selector,
    pooling_selector,
    probe_eval_radio,
    probe_init_selector,
    probe_pooling_radio,
    recon_slider,
    run_btn,
    seed_input,
    dataset_selector,
    init_selector,
    stage_selector,
    subjs_input,
    wandb_checkbox,
):
    # ── Block 2 view: only the parameters relevant to the selected stage ─────
    _stage = stage_selector.value
    _is_ocd = dataset_selector.value == "ocd"
    _rows = []
    # Pre-training run sits first, directly under the stage selector.
    if _stage == "stage3_2" and post_ckpt_selector.value is None:
        _rows.append(mo.callout(
            mo.md("No Pre-training run with a checkpoint was found. Run Stage 3.1 first."),
            kind="warn",
        ))
    if _stage == "stage3_2":
        _rows.append(mo.hstack([post_ckpt_selector], justify="start"))
    elif _stage == "dapt":
        _rows.append(mo.hstack([dapt_init_selector], justify="start"))
    elif _stage in ("stage1", "stage2"):
        _rows.append(mo.hstack([init_selector], justify="start"))

    # Shared parameters (no component).
    if _stage != "dapt":
        _top = [label_selector, mode_selector]
        if mode_selector.value == "Single Fold (CV)":
            _top.append(fold_slider)
        _rows.append(mo.hstack(_top, justify="start", wrap=True))
    # The OCD cohort always uses all subjects, so the scope control is not shown for it.
    _scope = [] if _is_ocd else [subjs_input]
    _rows.append(mo.hstack([epochs_slider, batch_size_dropdown, *_scope, seed_input],
                           justify="start", wrap=True))
    # Stage 3.1 only: its single optimizer LR, loss weights and output name.
    if _stage == "dapt":
        _rows.append(mo.hstack([lr_dropdown, dapt_weights_input, dapt_ckpt_name_input],
                               justify="start", wrap=True))

    # Component sub-blocks, in order: Encoder, Perceiver, Classifier.
    _encoder = []
    if _stage == "dapt":
        _encoder = [
            mo.md("**Mask Ratio**"),
            mo.hstack([dapt_mask_spatial_slider, dapt_mask_temporal_slider, dapt_mask_context_slider],
                      justify="start", wrap=True),
        ]
    elif _stage == "stage3_2":
        _encoder = [mo.hstack([encoder_lr_dropdown], justify="start")]
    _perceiver = [mo.hstack([backbone_lr_dropdown], justify="start")] if _stage in ("stage2", "stage3_2") else []
    _classifier = []
    if _stage != "dapt":
        _classifier = [mo.hstack([lr_dropdown, pooling_selector, dropout_slider, recon_slider],
                                 justify="start", wrap=True)]
    for _title, _items in (
        ("Encoder parameters", _encoder),
        ("Perceiver parameters", _perceiver),
        ("Classifier parameters", _classifier),
    ):
        if _items:
            _rows.append(mo.md(f"##### {_title}"))
            _rows.extend(_items)

    _rows.append(mo.hstack([wandb_checkbox, run_btn], justify="start"))

    # Stage 0 (logistic regression probe): discard every control that does not affect
    # the probe and show only what it uses.
    if _stage == "probe":
        _rows = [
            mo.md(
                "Logistic regression on **frozen encoder latents**. Encoders and Perceiver are "
                "not trained. Each window has 450 ROIs × 128 features; the pooling choice decides "
                "how these are reduced before the classifier:\n\n"
                "- **PCA (256 components):** no averaging. All 57,600 values are reduced to 256 "
                "principal components, fitted on each training split only. Slower and uses more "
                "memory.\n"
                "- **Per feature (mean over ROIs):** average over the 450 ROIs, leaving one 128-value "
                "whole-brain vector per window.\n"
                "- **Per ROI (mean over features):** average over the 128 features of each ROI, "
                "leaving one value per ROI (450 values).\n"
                "- **Per network (mean over ROIs per network):** average over the ROIs of each of the "
                "8 networks, leaving 8 × 128 = 1024 values."
            ),
            mo.hstack([label_selector, probe_init_selector], justify="start", wrap=True),
            mo.hstack([probe_pooling_radio], justify="start"),
            mo.hstack([probe_eval_radio], justify="start"),
            mo.hstack([*([] if _is_ocd else [subjs_input]), seed_input], justify="start", wrap=True),
            mo.hstack([run_btn], justify="start"),
        ]
        controls_accordion = mo.accordion({"⚙️ Logistic Regression Probe": mo.vstack(_rows)})
    else:
        controls_accordion = mo.accordion({"⚙️ Hyperparameters": mo.vstack(_rows)})
    controls_accordion
    return


@app.cell
def _(
    SimpleNamespace,
    backbone_lr_dropdown,
    batch_size_dropdown,
    clean_gpu_memory,
    dapt_ckpt_name_input,
    dapt_init_selector,
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
    post_ckpt_selector,
    pooling_selector,
    probe_eval_radio,
    probe_init_selector,
    probe_pooling_radio,
    recon_slider,
    run_btn,
    run_cross_validation,
    run_dapt,
    run_single_fold,
    run_single_split,
    dataset_selector,
    init_selector,
    seed_input,
    stage_selector,
    subjs_input,
    wandb_checkbox,
):
    # Outputs are always defined so downstream cells can read them before
    # training starts. Marimo compiles each cell body on its own, so an early
    # `return` is not allowed here; the work is gated with `if` instead.
    config, history, test_acc, test_f1 = None, None, None, None
    probe_payload = None

    # Stage 3.2 needs a DAPT checkpoint; without one, Start does nothing.
    _blocked = stage_selector.value == "stage3_2" and post_ckpt_selector.value is None
    if run_btn.value and not _blocked:
        clean_gpu_memory()

        is_cv_mode = mode_selector.value == "Full Cross-Validation (K-Fold)"
        is_fold_mode = mode_selector.value == "Single Fold (CV)"
        # Stage 0: logistic regression probe on frozen latents. It uses only label, encoder
        # init, evaluation mode, subjects and seed; all other controls are ignored.
        is_probe_mode = stage_selector.value == "probe"

        # ── Training stage → mode / freeze / encoder-lr / checkpoints ───────────
        stage = stage_selector.value
        is_dapt = stage == "dapt"
        freeze_perceiver = stage == "stage1"
        encoder_lr = float(encoder_lr_dropdown.value) if stage == "stage3_2" else 0.0
        # Stage 3.2 input: DAPT run checkpoint (path from the Previous Run dropdown).
        # Stage 1/2 input: optional DAPT run used as the starting weights (None = HCP-Aging).
        if stage == "stage3_2":
            dapt_ckpt = post_ckpt_selector.value
        elif stage in ("stage1", "stage2"):
            dapt_ckpt = init_selector.value
        else:
            dapt_ckpt = None
        # Stage 3.1 baseline: finetuned classifier checkpoint, or None = HCP-AGING init.
        dapt_init_ckpt = dapt_init_selector.value if is_dapt else None
        dataset_name = dataset_selector.value
        # OCD always uses all subjects; the scope control is hidden for it.
        n_subjs = None if dataset_name == "ocd" else subjs_input.value

        args = SimpleNamespace(
            dataset=dataset_name,
            label=label_selector.value,
            n_epochs=int(epochs_slider.value),
            batch_size=int(batch_size_dropdown.value),
            learning_rate=float(lr_dropdown.value),
            backbone_lr=float(backbone_lr_dropdown.value),
            freeze_perceiver=freeze_perceiver,
            n_subjs=n_subjs,
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
            dapt_init_ckpt=dapt_init_ckpt,
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
            patience=5,
        )

        try:
            import contextlib
            from OCD_symphony.model.finetuning import format_epoch_summary
            # Every bar opened for this run lives in _open and is closed in the finally below.
            _open = contextlib.ExitStack()
            _bars = {}
            _n_folds_cv = int(getattr(args, "n_folds", 5)) if is_cv_mode else 1
            _epoch_total = int(args.n_epochs) * _n_folds_cv
            if is_dapt:
                _train_title = "Stage 3.1: DAPT"
            elif is_cv_mode:
                _train_title = f"{_n_folds_cv}-Fold Cross-Validation"
            elif is_fold_mode:
                _train_title = "Single fold"
            else:
                _train_title = f"Training ({mode_selector.value})"

            def _on_load(ev):
                # Loading bar: one tick per file, shown while the dataset is read.
                if "load" not in _bars:
                    _bars["load"] = _open.enter_context(
                        mo.status.progress_bar(total=ev["total"], title="Loading dataset"))
                _bars["load"].update(increment=1)

            def _on_epoch(ep, m):
                # Epoch bar: one tick per finished epoch; subtitle carries the last epoch's metrics.
                if "epoch" not in _bars:
                    _bars["epoch"] = _open.enter_context(
                        mo.status.progress_bar(total=_epoch_total, title=_train_title))
                fold = m.get("fold")
                prefix = f"Fold {fold}/{_n_folds_cv} · " if fold is not None else ""
                epoch_no = m.get("epoch", ep)
                _bars["epoch"].update(
                    increment=1,
                    subtitle=f"{prefix}epoch {epoch_no}/{args.n_epochs} · {format_epoch_summary(m)}",
                )

            if is_probe_mode:
                spinner_msg = "Fitting logistic-regression probe on frozen latents..."
            elif is_dapt:
                spinner_msg = "Running DAPT (Stage 3.1) reconstruction pretraining..."
            elif is_cv_mode:
                spinner_msg = "Running 5-Fold Cross-Validation..."
            else:
                spinner_msg = f"Training Perceiver Classifier ({mode_selector.value})..."

            with mo.status.spinner(title=spinner_msg):
                if is_probe_mode:
                    # Imported lazily so a probe-side problem cannot break the whole notebook.
                    import contextlib
                    from OCD_symphony.model.linear_probe import run_logreg_probe as _run_probe
                    probe_args = SimpleNamespace(
                        dataset=dataset_name,
                        label=label_selector.value,
                        n_subjs=n_subjs,
                        seed=int(seed_input.value),
                        test_frac=0.15,
                        val_frac=0.15,
                        n_folds=5,
                        max_abs=1e4,
                        probe_eval=str(probe_eval_radio.value),
                        pooling=str(probe_pooling_radio.value),
                        dapt_ckpt=probe_init_selector.value,
                    )
                    # One progress bar per stage: extraction, then the split, or one per fold.
                    # Bars are created on a stage's first event and closed in the finally below.
                    _bars = {}
                    _stack = contextlib.ExitStack()

                    def _on_probe_event(ev, _bars=_bars, _stack=_stack, _n_folds=probe_args.n_folds):
                        if ev["stage"] == "extract":
                            key, title = "extract", "Extracting latents"
                        elif ev["stage"] == "split":
                            key, title = "split", "Logistic regression (split)"
                        else:
                            key = f"fold{ev['fold']}"
                            title = f"Fold {ev['fold'] + 1} / {_n_folds}"
                        if key not in _bars:
                            _bars[key] = _stack.enter_context(
                                mo.status.progress_bar(total=ev["total"], title=title))
                        if "done" in ev:
                            _bars[key].update(increment=1)

                    try:
                        probe_payload, _probe_path = _run_probe(probe_args, progress=_on_probe_event)
                    finally:
                        _stack.close()
                elif is_dapt:
                    # Stage 3.1 is a single reconstruction-pretraining run.
                    _ckpt_path, history = run_dapt(args, epoch_callback=_on_epoch, progress=_on_load)
                    test_acc, test_f1 = None, None
                elif is_cv_mode:
                    test_acc, test_f1, history = run_cross_validation(args, epoch_callback=_on_epoch, progress=_on_load)
                elif is_fold_mode:
                    test_acc, test_f1, history = run_single_fold(args, epoch_callback=_on_epoch, progress=_on_load)
                else:
                    test_acc, test_f1, history = run_single_split(args, epoch_callback=_on_epoch, progress=_on_load)
        finally:
            _open.close()
            clean_gpu_memory()

        config = None if is_probe_mode else vars(args)

    return config, history, test_acc, test_f1, probe_payload


@app.cell
def _(mo, probe_payload):
    # Logistic-regression probe summary (empty unless Stage 0 was run).
    _POOL_NAMES = {
        "pca": "PCA",
        "feature": "Per feature (mean over ROIs)",
        "roi": "Per ROI (mean over features)",
        "network": "Per network (mean over ROIs per network)",
    }

    def _header(_m, _model_features, _title):
        _pool = _m.get("pooling", "feature")  # legacy probe files predate pooling
        _pool_txt = _POOL_NAMES.get(_pool, _pool)
        if _pool == "pca":
            _pool_txt += f", {_model_features} components"
        else:
            _pool_txt += f", {_model_features} features"
        return (
            f"### Logistic-regression probe: {_title}\n"
            f"**Evaluation:** {_m.get('probe_eval', 'split')} · **Pooling:** {_pool_txt} · "
            f"**Encoders:** {_m['init']} · **Label:** {_m['label']} · Class weights: balanced\n\n"
        )

    def _regression_probe_view(payload):
        """Ridge probe (age): MAE / R² / r per split or per CV fold."""
        _m = payload["meta"]
        _pool_txt = _POOL_NAMES.get(_m.get("pooling"), _m.get("pooling"))
        _head = (
            f"### Ridge probe (age regression)\n"
            f"**Dataset:** {_m.get('dataset', 'hcp_ya')} · **Evaluation:** {_m.get('probe_eval')} · "
            f"**Pooling:** {_pool_txt} · **Encoders:** {_m['init']} · **Target:** {_m['label']}\n\n"
        )
        if "folds" in payload:
            _rows = [
                "| Fold | Test subjects | Best alpha | Test MAE | Test R² | Test r |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
            for _f in payload["folds"]:
                _t = _f["result"]["test"]
                _rows.append(
                    f"| {_f['fold'] + 1} | {_f['n_test_subjects']} | {_f['result']['best_alpha']:g} | "
                    f"{_t['mae']:.3f} | {_t['r2']:.3f} | {_t['r']:.3f} |"
                )
            _s = payload["summary"]
            _body = "\n".join(_rows) + (
                f"\n\n**Test MAE:** {_s['mae']['mean']:.3f} ± {_s['mae']['sd']:.3f} · "
                f"**Test R²:** {_s['r2']['mean']:.3f} ± {_s['r2']['sd']:.3f}"
            )
        else:
            _r = payload["result"]
            _rows = [
                "| Split | N windows | MAE | R² | r |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
            for _name in ("train", "val", "test"):
                _s = _r[_name]
                _rows.append(f"| {_name} | {_s['n']} | {_s['mae']:.3f} | {_s['r2']:.3f} | {_s['r']:.3f} |")
            _b = _r["mean_baseline"]
            _body = "\n".join(_rows) + (
                f"\n\nBest alpha: **{_r['best_alpha']:g}** (selected on validation MSE). "
                f"Predicting the training mean gives test MAE {_b['test_mae']:.3f}."
            )
        return mo.md(_head + _body + "\n\nSaved to `results/probes/`.")

    if probe_payload is None:
        _out = mo.md("")
    elif probe_payload["meta"].get("task") == "regression":
        _out = _regression_probe_view(probe_payload)
    elif "folds" in probe_payload:
        # Subject-level K-fold CV payload.
        _m = probe_payload["meta"]
        _sm = probe_payload["summary"]
        _n_feat = probe_payload["folds"][0]["result"]["n_features_model"]
        _rows = [
            "| Fold | Train subjects | Val subjects | Test subjects | Best C | Test Acc | Test Balanced Acc | Test Macro F1 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _f in probe_payload["folds"]:
            _t = _f["result"]["test"]
            _rows.append(
                f"| {_f['fold'] + 1} | {_f.get('n_train_subjects', '-')} | {_f.get('n_val_subjects', '-')} | "
                f"{_f['n_test_subjects']} | {_f['result']['best_C']:g} | "
                f"{_t['acc']:.3f} | {_t['balanced_acc']:.3f} | {_t['macro_f1']:.3f} |"
            )
        _ba, _ac = _sm["balanced_acc"], _sm["acc"]
        _total = _m.get("n_subjects", {}).get("total", _m.get("n_subjects_total"))
        _out = mo.md(
            _header(_m, _n_feat, f"{_m['n_folds']}-fold subject-level CV")
            + f"**Subjects:** {_total} total · C selected per fold on an inner subject-level "
            "validation split\n\n"
            + "\n".join(_rows)
            + f"\n\n**Test balanced accuracy:** {_ba['mean']:.3f} ± {_ba['sd']:.3f} (mean ± SD) · "
            f"**Test accuracy:** {_ac['mean']:.3f} ± {_ac['sd']:.3f}\n\n"
            f"Saved to `results/probes/`."
        )
    else:
        # Single subject-level split payload.
        _r = probe_payload["result"]
        _m = probe_payload["meta"]
        _ns = _m.get("n_subjects", {})
        _rows = [
            "| Split | Subjects | N windows | Acc | Balanced Acc | Macro F1 | Pred counts [c0, c1] |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _name in ("train", "val", "test"):
            _s = _r[_name]
            _rows.append(
                f"| {_name} | {_ns.get(_name, '-')} | {_s['n']} | {_s['acc']:.3f} | "
                f"{_s['balanced_acc']:.3f} | {_s['macro_f1']:.3f} | {_s['pred_counts']} |"
            )
        _b = _r["majority_baseline"]
        _out = mo.md(
            _header(_m, _r["n_features_model"], "single subject-level split")
            + f"**Subjects:** {_ns.get('total', _m['n_subjects_total'])} total · "
            f"Best C: **{_r['best_C']:g}** (selected on val log-loss)\n\n"
            + "\n".join(_rows)
            + f"\n\nMajority-class baseline (class {_b['majority_class']}): "
            f"val acc {_b['val_acc']:.3f}, test acc {_b['test_acc']:.3f}.\n\n"
            f"Saved to `results/probes/`."
        )
    _out
    return


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
                # Run settings live under meta.config; without them the stage cannot be labelled.
                config = {}
                try:
                    with open(path) as f:
                        payload = json.load(f)
                    config = (payload.get("meta") or {}).get("config") or payload.get("config") or {}
                except (OSError, ValueError):
                    config = {}
                runs.append({
                    "kind": "results",
                    "run_id": os.path.splitext(os.path.basename(path))[0],
                    "path": path,
                    "mtime": os.path.getmtime(path),
                    "config": config,
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
    # Current format: "[Epoch N] Val Loss: x | Val Acc: y | Val Bal Acc: z | Val F1: w | Time: t".
    # Older format: "[Epoch N] Val Acc: y | Val F1: w | ...". The Val Loss field is optional.
    _VAL_RE = re.compile(
        r"\[Epoch\s+(\d+)\]\s+(?:Val Loss:[^|\n]*\|\s*)?Val Acc:\s*([0-9.eE+-]+)[^\n]*?\|\s*Val F1:\s*([0-9.eE+-]+)"
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
            # Epochs without a parsed validation line are NaN (plotted as gaps), never 0.
            "val_acc": [val.get(e, (float("nan"), float("nan")))[0] for e in epochs],
            "val_f1": [val.get(e, (float("nan"), float("nan")))[1] for e in epochs],
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
        # Runs saved before the dataset option existed were all HCP-YA.
        parts.append(str(cfg.get("dataset", "hcp_ya")))
        for key in ("label", "pooling"):
            if cfg.get(key) not in (None, ""):
                parts.append(str(cfg[key]))
        if cfg.get("learning_rate") is not None:
            parts.append(f"lr{float(cfg['learning_rate']):.0e}")
        if cfg.get("n_epochs") is not None:
            parts.append(f"{cfg['n_epochs']}ep")
        if cfg.get("mode") == "dapt":
            parts.append("stage3.1")
        elif cfg.get("dapt_ckpt") and not cfg.get("freeze_perceiver"):
            # Stage 3.2: classifier run with a DAPT checkpoint and unfrozen Perceiver.
            parts.append("stage3.2")
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
        "dataset", "label", "mode", "n_epochs", "batch_size", "learning_rate", "backbone_lr",
        "encoder_lr", "freeze_perceiver", "pooling", "dropout", "recon_weight",
        "dapt_mask_ratio", "dapt_mask_ratio_spatial", "dapt_mask_ratio_temporal",
        "dapt_mask_ratio_context", "dapt_weights", "dapt_ckpt", "dapt_ckpt_name",
        "grad_clip", "max_abs", "val_mask_seed",
        "n_subjs", "n_folds", "fold_idx", "test_frac", "val_frac",
        "seed", "cv", "ddp", "use_wandb",
    ]
    _HYPERPARAM_DEFS = {
        "dataset": "Training dataset: hcp_ya (HCP-YA) or ocd (OCD cohort, all subjects).",
        "label": "Target: gender or age (HCP-YA); condition, gender or age (OCD). Age is a regression target.",
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
        "dapt_ckpt": "Stage 3.2 input: DAPT run checkpoint selected in the Pre-training run dropdown.",
        "dapt_init_ckpt": "Stage 3.1 baseline: finetuned classifier run whose Perceiver initialises DAPT (None = HCP-Aging baseline).",
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

        _is_reg = isinstance(history, dict) and history.get("task") == "regression"
        if _is_reg and is_cv:
            # CV summary for age regression: mean_acc = mean test MAE, mean_f1 = mean test R².
            _fold_hists = history.get("histories", [])
            cards = mo.hstack([
                mo.stat(
                    label="Mean Test MAE",
                    value=f"{history['mean_acc']:.3f} ± {history['std_acc']:.3f}",
                    caption=f"Years, across {len(history['fold_accs'])} CV folds",
                ),
                mo.stat(
                    label="Mean Test R²",
                    value=f"{history['mean_f1']:.3f} ± {history['std_f1']:.3f}",
                    caption="Leave-subject-out",
                ),
            ], justify="space-around")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            for f_idx, fold_hist in enumerate(_fold_hists):
                ep = list(range(1, len(fold_hist.get("train_loss", [])) + 1))
                if ep:
                    ax1.plot(ep, fold_hist["train_loss"], alpha=0.7, label=f"Fold {f_idx + 1}")
                    ax2.plot(ep, fold_hist["val_mae"], alpha=0.7, label=f"Fold {f_idx + 1}")
            ax1.set_title("Training Loss Across Folds", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss (MSE, standardised age)")
            ax2.set_title("Validation MAE Dynamics", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Val MAE (years)")
            for _ax in (ax1, ax2):
                _ax.grid(True, linestyle="--", alpha=0.6)
                _ax.legend()
            plt.tight_layout()
            return mo.vstack(prelude + [
                mo.md("### 📊 5-Fold Cross-Validation Summary (age regression)"),
                cards,
                mo.as_html(fig),
            ])

        if _is_reg:
            def _fmt(v, nd=3):
                return "N/A" if v is None or v != v else f"{v:.{nd}f}"

            cards = mo.hstack([
                mo.stat(label="Test MAE", value=_fmt(history.get("test_mae")),
                        caption="Years, held-out subjects"),
                mo.stat(label="Test R²", value=_fmt(history.get("test_r2"), 4),
                        caption="Held-out subject test set"),
                mo.stat(label="Subject MAE", value=_fmt(history.get("test_subj_mae")),
                        caption=f"n={history.get('test_subj_n')} subjects"),
                mo.stat(label="Epochs Completed", value=f"{len(history['train_loss'])}",
                        caption="Full iterations"),
            ], justify="space-around")
            epochs = list(range(1, len(history["train_loss"]) + 1))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            ax1.plot(epochs, history["train_loss"], marker="o", color="#1f77b4", label="Train Loss")
            ax1.plot(epochs, history["val_loss"], marker="s", color="#ff7f0e", label="Val Loss")
            ax1.set_title("Loss (MSE on standardised age)", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Epoch")
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend()
            ax2.plot(epochs, history["train_mae"], marker="o", color="#2ca02c", label="Train MAE")
            ax2.plot(epochs, history["val_mae"], marker="s", color="#ff7f0e", label="Val MAE")
            ax2.set_title("MAE (years)", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()
            plt.tight_layout()
            return mo.vstack(prelude + [
                mo.md("### 📊 Experiment Results (age regression)"),
                cards,
                mo.as_html(fig),
            ])

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
            # Subject-level accuracy (mean probability per subject). Absent in older runs.
            if history.get("train_subj_acc"):
                ax2.plot(epochs, [a * 100 for a in history["train_subj_acc"]], marker="o", linestyle="--",
                         color="#2ca02c", alpha=0.6, label="Train Subj Acc (%, train-mode)")
            if history.get("val_subj_acc"):
                ax2.plot(epochs, [a * 100 for a in history["val_subj_acc"]], marker="s", linestyle="--",
                         color="#ff7f0e", alpha=0.6, label="Val Subj Acc (%)")
            ax2.set_title("Accuracy & F1 Dynamics", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Score (%)")
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend()

            plt.tight_layout()

            _subj_test = history.get("test_subj_acc")
            _subj_line = (
                [mo.md(f"**Test subject accuracy:** {_subj_test * 100:.2f}% "
                       f"(n={history.get('test_subj_n')} subjects)")]
                if _subj_test is not None and _subj_test == _subj_test  # NaN check
                else []
            )

            dashboard_view = mo.vstack(prelude + [
                mo.md("### 📊 Experiment Results"),
                metric_cards,
                *_subj_line,
                mo.as_html(fig),
            ])

        return dashboard_view

    return (render_history,)


@app.cell
def _(config, history, mo, plt, render_history, test_acc, test_f1):
    _is_dapt = isinstance(history, dict) and history.get("is_dapt", False)
    _is_reg = isinstance(history, dict) and history.get("task") == "regression"
    mo.stop(
        test_acc is None and not _is_dapt and not _is_reg,
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
    # Weights used for reconstruction: the loaded DAPT checkpoint or the pre-trained HCP-Aging baseline.
    recon_source = mo.ui.dropdown(
        options={"DAPT checkpoint (loaded run)": "dapt", "Baseline (pre-trained on HCP-Aging)": "baseline"},
        value="DAPT checkpoint (loaded run)",
        label="Weights",
    )
    # Dataset whose subjects are reconstructed (and whose normalisation is used).
    recon_dataset = mo.ui.dropdown(
        options={"HCP-YA": "hcp_ya", "OCD": "ocd"},
        value="HCP-YA",
        label="Dataset",
    )
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
        recon_dataset,
        recon_fraction,
        recon_n_rois,
        recon_plot_width,
        recon_roi_strategy,
        recon_run_btn,
        recon_sample_idx,
        recon_source,
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
def _(find_ckpt, load_btn, mo, run_metas, run_selector, set_loaded_ckpt):
    # Store the checkpoint of the loaded run in state when Load is clicked.
    # A note is shown only when the run cannot be reconstructed because its encoder
    # weights are missing. The note is the cell's last expression (no early return:
    # marimo cells cannot return from inside a branch).
    _load_view = None
    if run_selector is not None and load_btn is not None and load_btn.value:
        from OCD_symphony.reconstruction.dapt_reconstruction import (
            MissingEncoderError, load_reconstruction_checkpoint,
        )
        _meta = run_metas.get(run_selector.value)
        if _meta is not None:
            _ck = find_ckpt(_meta)
            if _ck:
                try:
                    load_reconstruction_checkpoint(_ck, "cpu")
                except MissingEncoderError as _exc:
                    _load_view = mo.callout(mo.md(f"Cannot reconstruct this run: {_exc}"), kind="warn")
                except (ValueError, FileNotFoundError) as _exc:
                    print(f"[load] checkpoint not usable for reconstruction: {_exc}")
                else:
                    set_loaded_ckpt(_ck)
    _load_view



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
    recon_dataset,
    recon_run_btn,
    recon_source,
    set_recon_cache,
    set_recon_clicks,
):
    # ── Reconstruction: inference runs only when the button is clicked ───────
    from OCD_symphony.reconstruction.dapt_reconstruction import (
        default_device, estimate_seconds_per_subject, fit_normalisation_stats,
        reconstruct_subject_stats, select_test_subject_subset,
    )

    RECON_KEEP_K = 6  # best + worst + (K-2) median-scoring subjects kept in memory

    _source = recon_source.value
    _ds = recon_dataset.value
    _ds_name = "OCD" if _ds == "ocd" else "HCP-YA"
    # Cache slot per (dataset, weights): switching either one never reuses another result.
    _slot = f"{_ds}:{_source}"
    _dapt_ckpt = get_loaded_ckpt()
    _device = default_device()
    _title = mo.md("### 🧠 Reconstruction")
    recon_result = None

    # A click is "pending" when the counter is above the last processed value. Every pending
    # click is consumed here, in every branch, so it can never fire later when Weights change.
    _clicks = recon_run_btn.value
    _pending_click = _clicks > get_recon_clicks()
    if _pending_click:
        set_recon_clicks(_clicks)

    # Runnable: Baseline always, DAPT only once a run with a checkpoint is loaded.
    # The button is only shown when runnable, so it cannot be clicked otherwise (marimo
    # cannot toggle a button's disabled state without recreating it and resetting its counter).
    _runnable = not (_source == "dapt" and _dapt_ckpt is None)
    _controls_row = mo.hstack(
        [recon_dataset, recon_source, recon_fraction] + ([recon_run_btn] if _runnable else []),
        justify="start", wrap=True,
    )

    if not _runnable:
        recon_header = mo.vstack([
            _title,
            _controls_row,
            mo.md(
                "*Reconstruct BOLD is disabled: load a DAPT run with a saved checkpoint "
                "(**Load & Visualize History**) or choose **Baseline** above.*"
            ),
        ])
    else:
        _ckpt = _dapt_ckpt if _source == "dapt" else None
        _frac = recon_fraction.value / 100.0
        # Cold cache: the loading bar opens on the first file event; a warm cache emits nothing.
        import contextlib as _cl
        _stats_open = _cl.ExitStack()
        _stats_bars = {}

        def _on_stats_load(ev):
            if "load" not in _stats_bars:
                _stats_bars["load"] = _stats_open.enter_context(
                    mo.status.progress_bar(total=ev["total"], title="Preparing normalisation statistics"))
            _stats_bars["load"].update(increment=1)

        try:
            _stats = fit_normalisation_stats(n_subjs=500, dataset=_ds, progress=_on_stats_load)
        finally:
            _stats_open.close()
        _n_test_total = len(_stats["test_subjs"])
        _n_subj_req = max(1, round(_frac * _n_test_total))
        _eta_s = (_n_subj_req + RECON_KEEP_K) * estimate_seconds_per_subject(_device)
        _eta_txt = f"~{_eta_s:.0f} s" if _eta_s < 120 else f"~{_eta_s / 60:.1f} min"
        _weights = (
            ("DAPT checkpoint" if _source == "dapt" else "Baseline (pre-trained on HCP-Aging)")
            + f" on {_ds_name} data"
        )
        _key = (_slot, _ckpt, str(_device), recon_fraction.value)
        # One cached result per (dataset, weights): {slot: (key, result)}.
        _cache = get_recon_cache()

        if _pending_click:
            _subjs = select_test_subject_subset(_stats["test_subjs"], frac=_frac, seed=0)
            _total = len(_subjs) + RECON_KEEP_K
            with mo.status.progress_bar(
                total=_total,
                title=f"Reconstructing BOLD ({_weights})",
                subtitle=f"{len(_subjs)} subjects on {_device.type.upper()} (est. {_eta_txt})",
                completion_title="Reconstruction complete",
                completion_subtitle=f"{len(_subjs)} subjects",
            ) as _bar:
                _res = reconstruct_subject_stats(
                    _ckpt, _subjs, _stats["W"], _stats["B"],
                    keep_k=RECON_KEEP_K, device=_device,
                    progress=lambda done, total: _bar.update(),
                    dataset=_ds,
                )
            _res["fraction"] = recon_fraction.value
            _cache = {**_cache, _slot: (_key, _res)}
            set_recon_cache(_cache)

        _entry = _cache.get(_slot)
        recon_result = _entry[1] if (_entry is not None and _entry[0] == _key) else None
        if recon_result is None:
            _status = mo.md(
                f"👉 *Click **Reconstruct BOLD** to run the {_weights} on {recon_fraction.value}% of the "
                f"held-out test subjects ({_n_subj_req} subjects, est. {_eta_txt}).*"
            )
        else:
            _status = mo.md(
                f"✅ {_weights} reconstruction ready: **{recon_result['subjs'].shape[0]}** samples from "
                f"**{len(set(recon_result['subjs'].tolist()))}** subjects "
                f"({recon_fraction.value}% of {_n_test_total} test subjects). "
                f"Traces kept in memory for {len(recon_result['representative'])} representative subjects."
            )
        recon_header = mo.vstack([
            _title,
            _controls_row,
            mo.md(
                f"Device: **{_device.type.upper()}** · weights: **{_weights}** · selection: "
                f"**{recon_fraction.value}%** = **{_n_subj_req}** of {_n_test_total} test subjects · "
                f"estimated time **{_eta_txt}**"
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
    try:
        _result_runs = scan_result_runs()
        _all_runs = _result_runs + scan_wandb_runs()
    except Exception as _exc:
        # Never let run discovery block the controls: fall back to no previous runs.
        print(f"[run discovery] failed, continuing without previous runs: {_exc}")
        _all_runs = []
        _result_runs = []
    # The results JSON is always written at the end of a run and is the authoritative
    # history. Hide the W&B console entry of any run that already has a JSON (its
    # filename ends with the W&B id); the W&B parser is only a fallback for runs without one.
    _json_ids = {_r["run_id"] for _r in _result_runs}
    _all_runs = [
        _r for _r in _all_runs
        if _r["kind"] == "results"
        or not any(_j.endswith(f"_{_r['run_id']}") for _j in _json_ids)
    ]
    # Newest first (by file mtime). run_metas is a dict, so this order is the display order.
    _all_runs = sorted(_all_runs, key=lambda r: r.get("mtime") or 0, reverse=True)
    for _run in _all_runs:
        try:
            _meta = {
                "kind": _run["kind"],
                "run_id": _run["run_id"],
                "path": _run["path"],
                "mtime": _run["mtime"],
                "config": _run.get("config", {}),
                "summary": _run.get("summary", {}),
            }
            run_metas[run_label(_meta)] = _meta
        except Exception as _exc:
            print(f"[run discovery] skipped {_run.get('path')}: {_exc}")

    if run_metas:
        _ordered = list(run_metas.keys())
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
