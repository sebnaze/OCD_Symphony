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
    from types import SimpleNamespace
    from OCD_symphony.model.finetuning import (
        run_single_split,
        run_single_fold,
        run_cross_validation,
        clean_gpu_memory,
    )
    return (
        SimpleNamespace,
        clean_gpu_memory,
        mo,
        plt,
        run_cross_validation,
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
        stop=30,
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
        value=None,
        label="Dataset Scope",
    )
    seed_input = mo.ui.number(
        start=0,
        stop=9999,
        value=42,
        label="Random Seed",
    )
    wandb_checkbox = mo.ui.checkbox(
        value=False,
        label="Log run & model to Weights & Biases (W&B)",
    )
    run_btn = mo.ui.run_button(
        label="🚀 Start Fine-tuning",
        kind="success",
    )
    return (
        batch_size_dropdown,
        epochs_slider,
        fold_slider,
        label_selector,
        lr_dropdown,
        mode_selector,
        run_btn,
        seed_input,
        subjs_input,
        wandb_checkbox,
    )


@app.cell
def _(
    batch_size_dropdown,
    epochs_slider,
    fold_slider,
    label_selector,
    lr_dropdown,
    mo,
    mode_selector,
    run_btn,
    seed_input,
    subjs_input,
    wandb_checkbox,
):
    top_controls = [label_selector, mode_selector]
    if mode_selector.value == "Single Fold (CV)":
        top_controls.append(fold_slider)

    control_panel = mo.vstack([
        mo.hstack(top_controls, justify="start"),
        mo.hstack([epochs_slider, lr_dropdown, batch_size_dropdown, subjs_input, seed_input], justify="start"),
        mo.hstack([wandb_checkbox, run_btn], justify="start"),
    ])

    controls_accordion = mo.accordion({"⚙️ Experiment Configuration & Hyperparameters": control_panel})
    controls_accordion
    return


@app.cell
def _(
    SimpleNamespace,
    batch_size_dropdown,
    clean_gpu_memory,
    epochs_slider,
    fold_slider,
    label_selector,
    lr_dropdown,
    mo,
    mode_selector,
    run_btn,
    run_cross_validation,
    run_single_fold,
    run_single_split,
    seed_input,
    subjs_input,
    wandb_checkbox,
):
    mo.stop(not run_btn.value, mo.md("👉 *Configure parameters above and click **Start Fine-tuning** to run.* "))

    clean_gpu_memory()

    is_cv_mode = mode_selector.value == "Full Cross-Validation (K-Fold)"
    is_fold_mode = mode_selector.value == "Single Fold (CV)"

    args = SimpleNamespace(
        label=label_selector.value,
        n_epochs=int(epochs_slider.value),
        batch_size=int(batch_size_dropdown.value),
        learning_rate=float(lr_dropdown.value),
        n_subjs=subjs_input.value,
        n_folds=5,
        fold_idx=int(fold_slider.value) if is_fold_mode else None,
        test_frac=0.15,
        val_frac=0.15,
        seed=int(seed_input.value),
        use_wandb=bool(wandb_checkbox.value),
        recon_weight=1e-4,
        cv=is_cv_mode,
        ddp=False,
    )

    try:
        spinner_msg = (
            "Running 5-Fold Cross-Validation..."
            if is_cv_mode
            else f"Training Perceiver Classifier ({mode_selector.value})..."
        )
        with mo.status.spinner(title=spinner_msg):
            if is_cv_mode:
                test_acc, test_f1, history = run_cross_validation(args)
            elif is_fold_mode:
                test_acc, test_f1, history = run_single_fold(args)
            else:
                test_acc, test_f1, history = run_single_split(args)
    finally:
        clean_gpu_memory()

    return history, test_acc, test_f1


@app.cell
def _(history, mo, plt, test_acc, test_f1):
    mo.stop(test_acc is None)

    is_cv = isinstance(history, dict) and history.get("is_cv", False)

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
        for f_idx, h in enumerate(histories):
            if "train_loss" in h and len(h["train_loss"]) > 0:
                ep = list(range(1, len(h["train_loss"]) + 1))
                ax1.plot(ep, h["train_loss"], alpha=0.7, label=f"Fold {f_idx + 1}")
                ax2.plot(ep, [a * 100 for a in h["val_acc"]], alpha=0.7, label=f"Fold {f_idx + 1}")

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

        dashboard_view = mo.vstack([
            mo.md("### 📊 5-Fold Cross-Validation Summary"),
            metric_cards,
            table_md,
            mo.as_html(fig),
        ])
    else:
        metric_cards = mo.hstack([
            mo.stat(
                label="Test Accuracy",
                value=f"{test_acc * 100:.2f}%",
                caption="Held-out subject test set",
            ),
            mo.stat(
                label="Test Macro F1",
                value=f"{test_f1:.4f}",
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

        dashboard_view = mo.vstack([
            mo.md("### 📊 Experiment Results"),
            metric_cards,
            mo.as_html(fig),
        ])

    dashboard_view
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
