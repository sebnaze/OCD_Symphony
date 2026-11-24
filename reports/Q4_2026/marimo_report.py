# /// script
# dependencies = ["marimo==0.14.17", "numpy", "plotly==5.23.0"]
# ///
import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", css_file="report.css")


# ════════════════════════════════════════════════════════════════════════════
# Cell: Setup
# Purpose: imports and global plotly settings (white plot background).
# Depends on: nothing.
# Provides: mo, np, go, make_subplots, json, os.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _():
    import json
    import os

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    # White plot background for every figure (plotly's default template uses a grey plot area).
    pio.templates.default = "plotly_white"

    return go, json, make_subplots, mo, np, os


# ════════════════════════════════════════════════════════════════════════════
# Cell: Data loading
# Purpose: read the JSON result files from data/ next to the notebook (native run) or next to the page (html-wasm).
# Depends on: json, os, mo (Setup cell).
# Provides: PROBES (probe results), RUNS (training runs), RECON (reconstruction cases and atlas),
#           PROV (provenance), EXCL (exclusions).
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(json, mo, os):
    import sys

    # Location of the notebook: its folder when run natively, the page URL in html-wasm.
    # Relative (root-relative) paths are not used; data/ sits next to the notebook (native) or next
    # to the exported page (html-wasm, see the Pages workflow).
    _loc = mo.notebook_location()
    if _loc is None:
        raise RuntimeError("mo.notebook_location() is unavailable: cannot locate data/")

    # Native run/export: read the files from disk. In-browser (html-wasm, Pyodide), the files are
    # fetched over HTTP from the same site. In html-wasm, notebook_location() is the page URL, so a
    # page named *.html is reduced to its folder before appending data/.
    if sys.platform == "emscripten":
        from pyodide.http import open_url

        _base = str(_loc).rstrip("/")
        if _base.endswith(".html"):
            _base = _base.rsplit("/", 1)[0]

        def _load(name):
            return json.loads(open_url(f"{_base}/data/{name}").read())
    else:
        DATA_DIR = os.path.join(str(_loc), "data")

        def _load(name):
            with open(os.path.join(DATA_DIR, name)) as f:
                return json.load(f)

    PROBES = _load("probes.json")
    RUNS = _load("runs.json")
    RECON = _load("recon.json")
    PROV = _load("provenance.json")
    EXCL = _load("exclusions.json")
    return EXCL, PROBES, PROV, RECON, RUNS


# ════════════════════════════════════════════════════════════════════════════
# Cell: Shared constants and data access
# Purpose: labels, colours, cohort definitions, FC description, table formatting helpers,
#          headline-probe selection, training-run lookup (RUN_A/B/C), Section III run selection,
#          balanced-accuracy bar builder (fig_bacc) used by several sections.
# Depends on: PROBES, RECON, RUNS, go, np.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(PROBES, RECON, RUNS, go, np):
    CHANCE = 0.5
    BAR_HOVER = "%{fullData.name}<br>%{x}: %{y:.2f}<extra></extra>"
    COHORTS = (("hcp_499", "HCP-YA gender (499)"), ("ocd_97", "OCD condition (97)"))
    RECON_COHORTS = (("hcp", "HCP-YA"), ("ocd", "OCD"))
    # Model labels used throughout the report (the same names appear in tables and plots).
    MODEL_LABEL = {"baseline": "HCP-Aging latents", "A": "HCP-YA latents", "B": "HCP-YA + lower mask ratio",
                   "C": "HCP-YA + VICReg latents"}
    MODEL_COLOR = {"baseline": "#1f77b4", "A": "#ff7f0e", "B": "#2ca02c", "C": "#9467bd"}
    FC_ROI = ("FC (ROI-level)", "fc_full")
    FC_RSN = ("FC (RSN-level)", "fc_network")
    RSN_OF_ROI = RECON["atlas"]["rsn"]
    ROI_NAMES = RECON["atlas"]["roi_names"]
    NETS = list(dict.fromkeys(RSN_OF_ROI))
    # Display names for the resting-state networks (keys are the labels stored in the data files).
    NET_LABEL = {"sub_cort": "Subcortical", "Vis": "Visual", "SomMot": "Somato-Motor",
                 "DorsAttn": "Dorsal Attention", "SalVentAttn": "Salience Ventral Attention",
                 "Limbic": "Limbic", "Cont": "Control", "Default": "Default Mode"}

    FC_TEXT = (f"""
        For each subject, the windows are correlated ROI by ROI (450 Schaefer400 + Tian S3 
        ROIs), Fisher-z transformed and averaged, giving one 450 × 450 FC matrix per subject. \n
        
        **FC (ROI-level)** uses the upper triangle of this matrix (101,025 values, reduced to 256 PCA components 
        fitted on the training split). \n
        
        **FC (RSN-level)** averages the matrix within and between the 8 networks (7 RSNs and subcortical), giving 36 values. 
        No encoder is involved. These FC-only probes are the benchmark: a model is useful only if its latents beat purely FC features.
    """
    )

    def f3(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    def md_table(headers, rows):
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(lines)

    def canon(rows):
        return [r for r in rows if r.get("duplicate_of") is None]

    def headline_probe(cohort, ckpt):
        """Headline cell: feature pooling, target sample, subject protocol, CV5."""
        for r in canon(PROBES):
            if (r["probe"] == "logreg_pooled_latents" and r["cohort"] == cohort and r["ckpt"] == ckpt
                    and r["pooling"] == "feature" and r["target_scope"] == "sample"
                    and r["split_protocol"] == "subject" and r["probe_eval"] == "cv" and r["n_folds"] == 5):
                return r
        return None

    def fc_probe(cohort, feature_set):
        for r in canon(PROBES):
            if (r["probe"] == "logreg_subject_fc" and r["cohort"] == cohort and r["pooling"] == feature_set
                    and r["probe_eval"] == "cv"):
                return r
        return None

    def recon_case(cohort, model):
        return RECON["cases"].get(f"{cohort}_{model}")

    def run(source):
        for r in RUNS:
            if r["source"] == source:
                return r
        raise KeyError(source)

    RUN_A = run("gender_dapt_mS0.70_mT0.70_mC0.50_lr5e-05_dapt_uc58rue5.json")
    RUN_B = run("HCP_YA_gender_dapt_mS0.50_mT0.50_mC0.35_lr1e-05_from13e26a_dapt_5k0hx8ls.json")
    RUN_C = run("HCP_YA_gender_dapt_mS0.70_mT0.70_mC0.50_lr1e-05_vicreg0.01_from13e26a_dapt_ftyl9qho.json")

    def finetune_rows():
        out = []
        for r in RUNS:
            if (r["mode"] == "cls" and r["dataset"] == "hcp_ya" and r["label"] == "gender"
                    and r["n_subjs"] == 500 and not r["cv"] and r["test_acc"] is not None
                    and r["source"].lower().startswith(("hcp_ya_gender_stage", "gender_stage"))):
                if r["freeze_perceiver"]:
                    stage = "Stage 1 (classifier only)"
                elif (r["encoder_lr"] or 0) > 0:
                    stage = "Stage 3.2 (encoders + Perceiver + classifier)"
                else:
                    stage = "Stage 2 (Perceiver + classifier)"
                out.append({**r, "stage": stage})
        return sorted(out, key=lambda r: (r["stage"], r["source"]))

    # ── Balanced accuracy ────────────────────────────────────────────────────
    def probe_bacc(r):
        """(mean, sd or None) for a probe row. sd (error bar) only when the row has fold results."""
        if r is None:
            return None
        sd = r["balanced_acc_sd"] if r.get("fold_balanced_acc") else None
        return r["balanced_acc_mean"], sd

    def latent_getter(ckpt):
        return lambda cohort: probe_bacc(headline_probe(cohort, ckpt))

    def fc_getter(feature_set):
        return lambda cohort: probe_bacc(fc_probe(cohort, feature_set))

    def add_probe_bars(fig, name, cats, probes):
        ys = [None if p is None else p[0] for p in probes]
        errs = [0 if (p is None or p[1] is None) else p[1] for p in probes]
        fig.add_bar(name=name, x=cats, y=ys, error_y=dict(type="data", array=errs, visible=any(errs)),
                    hovertemplate=BAR_HOVER)

    def fig_bacc(series, height=293):
        """Single balanced-accuracy plot: one bar group per cohort, one bar per series (5-fold CV mean ± SD)."""
        fig = go.Figure()
        cats = [xl for _, xl in COHORTS]
        for name, get in series:
            add_probe_bars(fig, name, cats, [get(c) for c, _ in COHORTS])
        fig.add_hline(y=CHANCE, line_dash="dot")
        fig.update_layout(barmode="group", yaxis_title="Balanced Accuracy<br>(5-fold CV, mean ± SD)",
                          yaxis_range=[0, 1], height=height, margin=dict(t=30))
        return fig

    def rsn_r2_stats(case):
        r2 = np.asarray(case["r2_roi"], dtype=float)
        rsn = np.asarray(RSN_OF_ROI)
        means, sds = [], []
        for net in NETS:
            v = r2[rsn == net]
            means.append(float(v.mean()))
            sds.append(float(v.std(ddof=1)) if v.size > 1 else 0.0)
        return means, sds

    return (
        BAR_HOVER, CHANCE, COHORTS, FC_ROI, FC_RSN, FC_TEXT, MODEL_COLOR, MODEL_LABEL, NETS, RECON_COHORTS,
        ROI_NAMES, RSN_OF_ROI, RUN_A, RUN_B, RUN_C, f3, fc_getter, fig_bacc, headline_probe, latent_getter,
        md_table, recon_case, rsn_r2_stats, finetune_rows, canon, NET_LABEL,
    )


# ════════════════════════════════════════════════════════════════════════════
# Cell: Shared axis ranges
# Purpose: one axis range per quantity, reused by every reconstruction plot of that quantity
#          (ROI-wise R² histogram and per-RSN bars, Pearson R density, BOLD traces).
#          Loss-curve ranges are NOT shared; each loss plot sets its own (see the loss-curves cell).
# Depends on: RECON, RECON_COHORTS, NETS, rsn_r2_stats, np.
# Provides: R2_RANGE, R2_BINS, HIST_YMAX, PEARSON_RANGE, PEARSON_YMAX, BOLD_RANGE.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(NETS, RECON, RECON_COHORTS, np, rsn_r2_stats):
    REPORT_CASES = [RECON["cases"][f"{c}_{m}"] for c, _ in RECON_COHORTS for m in ("baseline", "A")]

    def _pad(lo, hi, frac=0.05):
        span = hi - lo
        return (lo - frac * span, hi + frac * span)

    _r2_all = []
    for _c in REPORT_CASES:
        _m, _s = rsn_r2_stats(_c)
        _r2_all += list(_c["r2_roi"]) + [a - b for a, b in zip(_m, _s)] + [a + b for a, b in zip(_m, _s)]
    R2_RANGE = (float(np.floor(min(_r2_all) * 20) / 20), float(np.ceil(max(_r2_all) * 20) / 20))
    R2_BINS = np.linspace(R2_RANGE[0], R2_RANGE[1], 21)
    HIST_YMAX = max(np.histogram(c["r2_roi"], bins=R2_BINS)[0].max() / len(c["r2_roi"]) for c in REPORT_CASES) * 1.05
    PEARSON_RANGE = (-1.0, 1.0)
    PEARSON_YMAX = max(max(d) for c in REPORT_CASES for d in c["corr_density"]["by_network"].values()) * 1.05
    BOLD_RANGE = _pad(min(min(min(v) for v in c["trace"]["orig"] + c["trace"]["recon"]) for c in REPORT_CASES),
                      max(max(max(v) for v in c["trace"]["orig"] + c["trace"]["recon"]) for c in REPORT_CASES))
    return BOLD_RANGE, HIST_YMAX, PEARSON_RANGE, PEARSON_YMAX, R2_BINS, R2_RANGE


# ════════════════════════════════════════════════════════════════════════════
# Cell: Reconstruction figures and tables
# Purpose: BOLD trace plots, ROI-wise R² distribution and per-RSN bars, Pearson R by network,
#          the cross-model R² comparison (Section II), and the per-dataset reconstruction block
#          (Sections I and II).
# Depends on: mo, go, make_subplots, np, shared constants, axis ranges, recon_case, rsn_r2_stats.
# Provides: recon_section(cohort, table_models, plot_model), fig_roi_r2_compare().
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(
    BAR_HOVER, BOLD_RANGE, HIST_YMAX, MODEL_COLOR, MODEL_LABEL, NETS, NET_LABEL, PEARSON_RANGE, PEARSON_YMAX, R2_BINS,
    R2_RANGE, RECON, RECON_COHORTS, ROI_NAMES, RSN_OF_ROI, f3, go, make_subplots, md_table, mo, np, recon_case,
    rsn_r2_stats,
):
    # ── Reconstruction figures ───────────────────────────────────────────────
    def fig_trace(cohort, model):
        tr = recon_case(cohort, model)["trace"]
        rois = tr["rois"]
        t0, t1 = tr["t_window"]
        t = list(range(t0, t1))
        titles = [f"{ROI_NAMES[i]} ({RSN_OF_ROI[i]}) · R={r:.3f}" for i, r in zip(rois, tr["r"])]
        fig = make_subplots(rows=len(rois), cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=titles)
        for k, (orig, rec) in enumerate(zip(tr["orig"], tr["recon"])):
            fig.add_scatter(x=t, y=orig, mode="lines", line=dict(color="black", width=1), name="original",
                            showlegend=(k == 0), row=k + 1, col=1)
            fig.add_scatter(x=t, y=rec, mode="lines", line=dict(color="red", width=1, dash="dash"),
                            name="reconstructed", showlegend=(k == 0), row=k + 1, col=1)
        fig.update_annotations(font_size=11)
        fig.update_yaxes(range=BOLD_RANGE)
        fig.update_xaxes(title_text="TR", row=len(rois), col=1)
        fig.update_layout(height=round((110 * len(rois) + 90) * 2 / 3), margin=dict(t=40, b=40))
        return fig

    def fig_roi_r2_dist(cohort, model):
        c = recon_case(cohort, model)
        counts, _ = np.histogram(c["r2_roi"], bins=R2_BINS)
        centers = (R2_BINS[:-1] + R2_BINS[1:]) / 2
        fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6],
                            subplot_titles=("ROI-wise R² distribution", "ROI-wise R² per RSN (mean ± SD)"))
        fig.update_annotations(font_size=11)
        fig.add_bar(x=centers, y=counts / len(c["r2_roi"]), width=R2_BINS[1] - R2_BINS[0], name="ROIs",
                    marker_color=MODEL_COLOR[model], row=1, col=1,
                    hovertemplate="R² ≈ %{x:.2f}<br>probability %{y:.2f}<extra></extra>")
        means, sds = rsn_r2_stats(c)
        fig.add_bar(name="RSN", x=[NET_LABEL[n] for n in NETS], y=means, error_y=dict(type="data", array=sds),
                    marker_color=MODEL_COLOR[model], row=1, col=2, hovertemplate=BAR_HOVER)
        fig.update_xaxes(title_text="ROI-wise R²", range=R2_RANGE, row=1, col=1)
        fig.update_yaxes(title_text="probability", range=[0, HIST_YMAX], row=1, col=1)
        fig.update_yaxes(title_text="R²", range=R2_RANGE, row=1, col=2)
        fig.update_layout(height=227, showlegend=False, margin=dict(t=40))
        return fig

    def fig_pearson_by_network(cohort, model):
        d = recon_case(cohort, model)["corr_density"]
        fig = go.Figure()
        for net, dens in d["by_network"].items():
            fig.add_scatter(x=d["centers"], y=dens, mode="lines", name=NET_LABEL[net],
                            hovertemplate="%{fullData.name}<br>r ≈ %{x:.2f}<br>density %{y:.2f}<extra></extra>")
        fig.update_layout(xaxis_title="Pearson R (per window, per ROI)", yaxis_title="density", height=227,
                          margin=dict(t=30), xaxis_range=list(PEARSON_RANGE), yaxis_range=[0, PEARSON_YMAX])
        return fig

    def fig_roi_r2_compare():
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.14,
                            subplot_titles=[name for _, name in RECON_COHORTS])
        fig.update_annotations(font_size=11)
        for col, (cohort, _) in enumerate(RECON_COHORTS, start=1):
            for model in ("baseline", "A"):
                means, sds = rsn_r2_stats(recon_case(cohort, model))
                fig.add_bar(name=MODEL_LABEL[model], x=[NET_LABEL[n] for n in NETS], y=means,
                            error_y=dict(type="data", array=sds),
                            marker_color=MODEL_COLOR[model], hovertemplate=BAR_HOVER, row=1, col=col,
                            showlegend=(col == 1))
        fig.update_yaxes(title_text="ROI-wise R²<br>(mean ± SD over ROIs in RSN)", range=R2_RANGE, row=1, col=1)
        fig.update_yaxes(title_text=None, range=R2_RANGE, row=1, col=2)
        fig.update_layout(barmode="group", height=267, margin=dict(t=40))
        return fig

    # ── Tables (markdown) ────────────────────────────────────────────────────
    def dataset_title(cohort):
        c = recon_case(cohort, "baseline")
        return f"{dict(RECON_COHORTS)[cohort]} (test set, {c['n_subjects']} subjects, {c['n_windows']} windows)"

    def recon_section(cohort, table_models, plot_model):
        """Per dataset: header, reconstruction table, traces, ROI-wise R² distribution, Pearson R by network."""
        tr = recon_case(cohort, plot_model)["trace"]
        rows = []
        for m in table_models:
            c = recon_case(cohort, m)
            rows.append([MODEL_LABEL[m], f3(c["r2_roi_mean"]), f3(c["r2_roi_median"]),
                         f"{f3(c['corr_sample_mean'])} [{f3(c['corr_sample_sd'])}]"])
        return mo.vstack([
            mo.md(f"#### {dataset_title(cohort)}"),
            mo.md(md_table(["Model", "R² mean", "R² median", "Pearson R mean [S.D.]"], rows)),
            mo.md(f"*Illustration only.* One subject, `{tr['subject']}` ({MODEL_LABEL[plot_model]}), one ROI per RSN. "
                  "The trace shows what a reconstruction looks like; it is not a result."),
            mo.md("**BOLD traces.** Original (black) and reconstructed (red, dashed) signal of one representative "
                  "subject, one ROI per network. Look for the red trace following the slow fluctuations of the black "
                  "trace; systematic amplitude or timing mismatches mean signal is lost."),
            fig_trace(cohort, plot_model),
            mo.md("**ROI-wise R² distribution (left) and per network (right).** Left: how well each ROI is "
                  "reconstructed, across test subjects. Right: mean ± SD of R² per resting-state network. Look for "
                  "mass shifted towards higher R², and for networks with low means or large spread."),
            fig_roi_r2_dist(cohort, plot_model),
            mo.md("**Pearson R by network.** Density of window-level correlations between original and reconstructed "
                  "signal, per network. Look for distributions shifted to the right (higher r) and narrow spread."),
            fig_pearson_by_network(cohort, plot_model),
        ])

    return fig_roi_r2_compare, recon_section


# ════════════════════════════════════════════════════════════════════════════
# Cell: Classification figures and tables
# Purpose: linear-probe tables (probe_md, dapt_compare_md), Section III finetuning table and chart,
#          and the Section 1.3 FC-vs-latent balanced-accuracy chart.
# Depends on: mo, go, CHANCE, MODEL_*, FC_*, headline_probe, fc_getter, latent_getter, fig_bacc,
#             finetune_rows, f3, md_table.
# Provides: fig_fc(), fig_finetune(), probe_md(ckpts), dapt_compare_md(), finetune_md().
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(
    CHANCE, FC_ROI, FC_RSN, MODEL_COLOR, MODEL_LABEL, f3, fc_getter, fig_bacc, go, headline_probe,
    latent_getter, md_table, finetune_rows,
):
    def fig_fc():
        return fig_bacc([(MODEL_LABEL["baseline"], latent_getter("baseline")),
                         (FC_ROI[0], fc_getter(FC_ROI[1])), (FC_RSN[0], fc_getter(FC_RSN[1]))], height=280)

    def fig_finetune():
        rows = finetune_rows()
        n = len(rows)
        ckpt_name = {"baseline": "HCP-Aging", "A": "HCP-YA"}
        pool_shape = {"flatten": "/", "attention": "+", "mean": "x"}
        fig = go.Figure()
        fig.add_bar(
            orientation="h", y=list(range(n)), x=[r["test_acc"] for r in rows],
            customdata=[[f"{r['learning_rate']:g}", f"{r['encoder_lr']:g}"] for r in rows],
            marker=dict(color=[MODEL_COLOR.get(r["dapt_ckpt"], "#7f7f7f") for r in rows],
                        pattern=dict(shape=[pool_shape.get(r["pooling"], "") for r in rows], fillmode="overlay",
                                     solidity=0.3, fgcolor="#222")),
            hovertemplate="Test accuracy: %{x:.3f}<br>LR: %{customdata[0]}<br>Encoder LR: %{customdata[1]}<extra></extra>",
            showlegend=False,
        )
        # Legend-only entries. Colour (encoder checkpoint) in legend 1, fill (pooling) in legend 2.
        for ckpt in ("baseline", "A"):
            fig.add_bar(orientation="h", x=[], y=[], name=ckpt_name[ckpt], marker_color=MODEL_COLOR[ckpt],
                        legend="legend")
        for pool, shp in pool_shape.items():
            fig.add_bar(orientation="h", x=[], y=[], name=pool, legend="legend2",
                        marker=dict(color="#bbbbbb", pattern=dict(shape=shp, fillmode="overlay", solidity=0.3,
                                                                 fgcolor="#222")))
        fig.add_vline(x=CHANCE, line_dash="dot")
        fig.update_layout(
            barmode="overlay",
            xaxis=dict(title="test accuracy (single split)", range=[0, 1]),
            yaxis=dict(tickmode="array", tickvals=list(range(n)), ticktext=[r["stage"] for r in rows],
                       autorange="reversed"),
            height=round((150 + 52 * n) * 2 / 3), margin=dict(l=10, r=20, t=30, b=60),
            legend=dict(title=dict(text="Encoder ckpt"), x=0.99, xanchor="right", y=0.99, yanchor="top",
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#cccccc", borderwidth=1),
            legend2=dict(title=dict(text="Pooling"), x=0.99, xanchor="right", y=0.55, yanchor="top",
                         bgcolor="rgba(255,255,255,0.9)", bordercolor="#cccccc", borderwidth=1),
        )
        return fig

    def probe_md(ckpts):
        rows = []
        for cohort, cname in (("hcp_499", "HCP-YA gender, 499"), ("ocd_97", "OCD condition, 97")):
            for ckpt in ckpts:
                r = headline_probe(cohort, ckpt)
                if r is None:
                    rows.append([cname, MODEL_LABEL[ckpt], "MISSING", "", ""])
                    continue
                rows.append([cname, MODEL_LABEL[ckpt], f"{f3(r['balanced_acc_mean'])} [{f3(r['balanced_acc_sd'])}]",
                             f3(r["acc_mean"]), " / ".join(f3(v) for v in r["fold_balanced_acc"])])
        return md_table(["Cohort", "Model", "Balanced Accuracy (5-fold CV)", "Acc", "Folds"], rows)

    def dapt_compare_md():
        rows = []
        for c, cname in (("hcp_499", "HCP-YA 499"), ("ocd_97", "OCD 97")):
            b, a = headline_probe(c, "baseline"), headline_probe(c, "A")
            rows.append([cname,
                         f3(b["balanced_acc_mean"]) if b else "MISSING",
                         f3(a["balanced_acc_mean"]) if a else "MISSING",
                         f"{a['balanced_acc_mean'] - b['balanced_acc_mean']:+.3f}" if (a and b) else "n/a"])
        return md_table(["Cohort", "Baseline Balanced Accuracy", "A Balanced Accuracy", "Δ (A − baseline)"], rows)

    def finetune_md():
        rows = []
        for r in finetune_rows():
            ckpt = "pretrained" if r["dapt_ckpt"] == "A" else r["dapt_ckpt"]
            rows.append([r["stage"], ckpt, r["pooling"], f"{r['learning_rate']:g}",
                         f"{r['encoder_lr']:g}", r["split_protocol"] or "subject",
                         "LEAKY (window-level split)" if r["leaky"] else "", f3(r["test_acc"]), f3(r["test_f1"])])
        return md_table(["Stage", "Encoder ckpt", "Pooling", "LR", "Encoder LR", "Split", "Note",
                         "Test acc (single split)", "Test F1"], rows)

    return fig_fc, fig_finetune, probe_md, dapt_compare_md, finetune_md


# ════════════════════════════════════════════════════════════════════════════
# Cell: Loss curves
# Purpose: per-epoch training/validation loss plots for runs A, B and C. Each plot sets its own
#          y-range from its own data (not shared across plots). Epoch x-range is shared.
# Depends on: RUN_A, RUN_B, RUN_C.
# Provides: fig_dapt_curves(), fig_b_curves(), fig_c_curves().
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(RUN_A, RUN_B, RUN_C, go):
    _LOSS_KEYS = ("train_recon", "val_recon", "train_vicreg_total", "val_vicreg_total")
    LOSS_EPOCHS = (1, max(len(run_row["history"]["train_recon"]) for run_row in (RUN_A, RUN_B, RUN_C)))

    def _pad(lo, hi, frac=0.05):
        span = hi - lo
        return (lo - frac * span, hi + frac * span)

    def _curve_fig(run_row, keys, title):
        fig = go.Figure()
        vals_all = []
        for k in keys:
            if k in run_row["history"]:
                vals = run_row["history"][k]
                vals_all += list(vals)
                fig.add_scatter(x=list(range(1, len(vals) + 1)), y=vals, mode="lines+markers", name=k)
        # Individual y-range for this plot only (padded min/max of the plotted values).
        y_lo, y_hi = _pad(min(vals_all), max(vals_all))
        fig.update_layout(title=title, xaxis_title="epoch", yaxis_title="loss", height=240, margin=dict(t=40),
                          xaxis_range=list(LOSS_EPOCHS), yaxis_range=[y_lo, y_hi])
        return fig

    def fig_dapt_curves():
        return _curve_fig(RUN_A, ["train_recon", "val_recon"], "A: DAPT MSE (uc58rue5), 30 epochs")

    def fig_b_curves():
        return _curve_fig(RUN_B, ["train_recon", "val_recon"], "B: lower mask ratio, MSE, 20 epochs")

    def fig_c_curves():
        return _curve_fig(RUN_C, ["train_recon", "val_recon", "train_vicreg_total", "val_vicreg_total"],
                          "C: DAPT VICReg, 20 epochs")

    return fig_b_curves, fig_c_curves, fig_dapt_curves


# ════════════════════════════════════════════════════════════════════════════
# Cell: Introduction
# Purpose: "Introduction" page: motivation, data description, overview diagram and the list of
#          analysis sections (I–IV). Edit the markdown directly here.
# Depends on: mo.
# Provides: intro.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    import os as _os
    import sys as _sys

    # Same location rule as the data cell: the notebook folder (native) or the page URL (html-wasm).
    _loc = mo.notebook_location()
    if _loc is None:
        raise RuntimeError("mo.notebook_location() is unavailable: cannot locate figures/")
    _IMG = "brain_symphony_overview_diagram.jpeg"
    if _sys.platform == "emscripten":
        _base = str(_loc).rstrip("/")
        if _base.endswith(".html"):
            _base = _base.rsplit("/", 1)[0]
        _overview = f"{_base}/figures/{_IMG}"  # URL: the browser loads it from the site
    else:
        _overview = _os.path.join(str(_loc), "report_marimo", "figures", _IMG)
        if not _os.path.isfile(_overview):
            raise FileNotFoundError(f"Overview figure not found: {_overview}")

    intro = mo.vstack([
        mo.md(f"""
    ## Introduction

    **Goal.** Use the Brain Symphony foundation model (https://arxiv.org/pdf/2506.18314) to classify OCD vs healthy control and predict symptom severity (YBOCS scores).

    **Problem.** The model was pre-trained on HCP-Aging with a different preprocessing pipeline from ours. Our OCD cohort (52 OCD and 45 age- and gender-matched controls, 97 in total, 12 minutes of resting-state fMRI each) is too small to retrain the 5.6M-parameter model.

    **Approach.** We adapt the model with domain-adaptive pre-training (DAPT) on HCP-Young Adults (HCP-YA): 1066 healthy subjects, two 15-minute resting-state sessions each, preprocessed similarly to our OCD data.
    """),
        mo.image(src=_overview, alt="Brain Symphony overview diagram", width="90%"),
        mo.md(f"""

    **Roadmap**

    - **I. Baseline analysis.** Reconstruction and linear probes with the HCP-Aging weights (1.1, 1.2), plus FC-only benchmarks (1.3).
    - **II. Pre-training (DAPT).** Reconstruction and classification after DAPT, compared with Section I (2.1, 2.2).
    - **III. Post-training.** Supervised finetuning for HCP-YA gender, comparing three training stages (3.1). Triggered because DAPT did not improve classification.
    - **IV. Extras.** A VICReg regularizer (4.1) and a lower mask ratio (4.2), tested during DAPT.

    """),
    ])
    return (intro,)


# ════════════════════════════════════════════════════════════════════════════
# Cell: Section I — Baseline analysis
# Purpose: original (HCP-Aging) encoders: reconstruction (1.1), linear probes on latents (1.2),
#          and probes on FC features (1.3).
# Depends on: mo, recon_section, probe_md, fig_fc, FC_TEXT, RECON_COHORTS.
# Provides: section_1.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(FC_TEXT, RECON_COHORTS, fig_fc, mo, probe_md, recon_section):
    section_1 = mo.vstack([
        mo.md(f"""
            ## I. Baseline analysis\n
            
            **Purpose.** Before any training, we measured how well the pretrained Brain Symphony encoders, 
            using the HCP-Aging weights, represent our data. These results are the reference for every later 
            section.\n\n
            
            **Methods.** The encoders were trained to reconstruct masked BOLD time series. We report 
            reconstruction quality as ROI-wise R² and as the window-level Pearson correlation between original and 
            reconstructed signal, on the test subjects of each dataset. For classification we fitted logistic-regression 
            linear probes on the frozen latents and evaluated them with 5-fold cross-validation at the subject level, 
            using balanced accuracy (chance = 0.50). As a benchmark we fitted the same classifiers on functional 
            connectivity (FC) features computed directly from the BOLD data, without an encoder.\n\n

            **Why.** Reconstruction shows whether the encoders can model our data, which differs from the HCP-Aging 
            preprocessing. The FC benchmark defines what a useful latent representation must beat: if FC features 
            alone classify as well as the latents, the latents add no information.\n\n
            
            **Preliminary result.** With the HCP-Aging weights, reconstruction is poor (mean ROI-wise R² 0.034 for 
            HCP-YA and 0.075 for OCD). Latent probes reach 0.54 for HCP-YA gender and 0.46 for OCD condition, the 
            latter below chance. FC-only probes reach 0.61 for gender and 0.59 to 0.67 for OCD condition, so the 
            baseline latents are weaker than FC features.
            """),
        mo.md("### 1.1 Reconstruction (HCP-Aging latents)"),
        *[recon_section(c, ("baseline",), "baseline") for c, _ in RECON_COHORTS],
        mo.md("Note that despite not retraining the model, the traces shown here are illustrative only: one subject per cohort, for comparison of the reconstruction with the original signal."),
        mo.md("### 1.2 Linear probes on original latents"),
        mo.md(probe_md(("baseline",))),
        mo.md("### 1.3 Probes on original latents and FC (subject level)\n" + FC_TEXT +
              "\n\nNote that FC has one matrix per subject, so there is no window-level (sample) variant. "),
        mo.md("**Balanced accuracy, latents vs FC.** 5-fold CV balanced accuracy (mean ± SD) for the HCP-Aging "
              "latents and the two FC feature sets, per cohort. The dotted line is chance (0.50). Look for whether the "
              "latent bar reaches the FC bars; bars below the FC benchmark mean the latents add no information beyond FC."),
        fig_fc(),
        mo.md("Error bars: 5-fold CV SD.")
    ])
    return (section_1,)


# ════════════════════════════════════════════════════════════════════════════
# Cell: Section II — Pre-Training (DAPT)
# Purpose: DAPT checkpoint A: reconstruction (2.1), cross-model R² comparison, DAPT loss curve,
#          classification (2.2) and comparison with Section I (2.3).
# Depends on: mo, recon_section, fig_roi_r2_compare, fig_dapt_curves, probe_md, fig_bacc,
#             fc_getter, latent_getter, FC_ROI, MODEL_LABEL, RECON_COHORTS, dapt_compare_md.
# Provides: section_2.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(
    FC_ROI, MODEL_LABEL, RECON_COHORTS, fc_getter, fig_bacc, fig_dapt_curves, fig_roi_r2_compare, latent_getter,
    mo, probe_md, recon_section, dapt_compare_md,
):
    section_2 = mo.vstack([
        mo.md(f"""
            ## II. Pre-training\n

            **Purpose.** The HCP-Aging encoders were trained on data with a different preprocessing pipeline, and 
            our OCD cohort (97 subjects) is too small to retrain a 5.6M-parameter model. We therefore performed 
            domain-adaptive pre-training (DAPT): masked-reconstruction training of the encoders on HCP-Young Adults 
            BOLD data, which uses a preprocessing pipeline similar to ours. The 499 HCP-YA subjects of the gender 
            cohort were seen by the encoders without labels; labels were not used in this stage.\n\n
            
            **Methods.** This DAPT pretraining run (checkpoint A) was trained with the MSE reconstruction loss for 30 epochs (learning rate 
            5e-5, mask ratios 0.7 / 0.7 / 0.5 for spatial, temporal and context masking). The reconstruction and 
            probe evaluation of Section I were repeated for this checkpoint on the same subjects.\n\n

            **Why.** DAPT should adapt the latents to the HCP-YA and OCD signal statistics. Reconstruction is the 
            direct test of that adaptation; the probes test whether the adapted latents carry more diagnostic or 
            demographic information.\n\n

            **Preliminary result.** Reconstruction improves substantially on both cohorts (mean ROI-wise R² 0.466 
            for HCP-YA and 0.473 for OCD, against 0.034 and 0.075 in Section I). Classification changes little: 
            gender balanced accuracy moves from 0.535 to 0.557, and OCD from 0.462 to 0.481. Both changes are 
            within the 5-fold CV standard deviations (about 0.02 for gender and 0.13 to 0.15 for OCD), so DAPT 
            did not measurably improve classification.
            """),
        mo.md("### 2.1 Reconstruction (checkpoint A)\n"),
        mo.accordion({"Show loss curve": mo.vstack([
            mo.md(f"""**Training loss.** Reconstruction loss per epoch for pretraining (Checkpoint A, MSE, 30 epochs), on training and validation data. Look for validation loss falling together with training loss and no widening gap between them; a widening gap would indicate overfitting to the training subjects."""),
            fig_dapt_curves()
        ])}),
        *[recon_section(c, ("baseline", "A"), "A") for c, _ in RECON_COHORTS],
        mo.md("**ROI-wise R², HCP-Aging vs HCP-YA latents.** Mean ± SD of ROI-wise R² per network, for the "
              "HCP-Aging weights (blue) and checkpoint A (orange), on each cohort. Look for orange bars above blue "
              "bars in every network. A uniform gain suggests DAPT adapted the whole model; a gain confined to a few "
              "networks shows which networks the adaptation helped."),
        fig_roi_r2_compare(),
        mo.md("### 2.2 Classification"),
        mo.md(probe_md(("baseline", "A"))),
        mo.md("**Balanced accuracy after DAPT.** FC benchmark, HCP-Aging latents (blue) and checkpoint A (orange), "
              "per cohort, with dotted chance line. Look for an orange bar exceeding the blue bar by more than the error "
              "bars; a difference inside the error bars is not an improvement."),
        fig_bacc([(FC_ROI[0], fc_getter(FC_ROI[1])), (MODEL_LABEL["baseline"], latent_getter("baseline")),
                  (MODEL_LABEL["A"], latent_getter("A"))]),
        mo.md("FC (ROI-level) is the FC-only benchmark (see Section 1.3)."),
        mo.md("### 2.3 Comparison with Section I (same cohort and protocol)"),
        mo.md(dapt_compare_md()),
        mo.md("Reconstruction deltas are in the 2.1 table (identical test subjects per cohort)."),
    ])
    return (section_2,)


# ════════════════════════════════════════════════════════════════════════════
# Cell: Section III — Post-training (finetuning)
# Purpose: finetuning on HCP-YA gender: table and chart of all qualifying single-split runs.
# Depends on: mo, finetune_md, fig_finetune.
# Provides: section_3.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(fig_finetune, mo, finetune_md):
    section_3 = mo.vstack([
        mo.md(f"""
            ## III. Post-training\n

            **Purpose.** Section II showed that DAPT improves reconstruction but not classification. We therefore 
            tested whether supervised finetuning on HCP-YA gender, with labels, lets the encoders support the 
            classification task. HCP-YA gender labels are available for this cohort, so gender serves as the 
            proof-of-concept task.\n\n

            **Methods.** Three training stages were compared: (1) classifier only, with encoders and Perceiver frozen; 
            (2) Perceiver and classifier trained, encoders frozen; (3) encoders, Perceiver and classifier trained 
            together. Pooling of the latent sequence was flatten, attention or mean, and learning rates were varied. 
            Each run used one train/validation/test split of 500 subjects.\n\n

            **Why.** Finetuning tests whether the latents contain task-relevant information that a frozen probe 
            cannot access.\n\n
            
            **Preliminary result.** Single-split test accuracy ranges from 0.49 to 0.63. The best run (attention 
            pooling, encoders updated) reaches 0.63. A single-split FC-only probe on the same cohort reached 0.68, 
            higher than every finetuned run. The split protocols differ, so this comparison is indicative only. 
            Because each configuration was evaluated on one split without cross-validation, differences of a few 
            points between runs are within expected variation.
            """),
        mo.md("### 3.1 Finetuning runs\n"
              "**Single-split test accuracy per configuration.** Each bar is one run; colour gives the encoder "
              "checkpoint, fill pattern the pooling, and the dotted line is chance. Hover for learning rates. Look for "
              "runs clearly above chance, and whether the stage that updates the encoders (Stage 3) beats the frozen-"
              "encoder stages (Stages 1 and 2)."),
        mo.md(finetune_md()),
        fig_finetune(),
    ])
    return (section_3,)


# ════════════════════════════════════════════════════════════════════════════
# Cell: Section IV — Extras (VICReg and lower mask ratio)
# Purpose: VICReg run C vs MSE run A (4.1), lower mask ratio run B (4.2), with loss curves
#          and the balanced-accuracy summary.
# Depends on: mo, probe_md, fig_c_curves, fig_b_curves, fig_bacc, fc_getter, latent_getter,
#             FC_ROI, MODEL_LABEL.
# Provides: section_4.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(FC_ROI, MODEL_LABEL, fc_getter, fig_b_curves, fig_bacc, fig_c_curves, latent_getter, mo, probe_md):
    section_4 = mo.vstack([
        mo.md(f"""
            ## IV. Extras\n

            **Purpose.** Two variants of DAPT were tested to see whether the representation or the reconstruction 
            can be improved further. (a) A VICReg regularizer (variance, invariance, covariance terms) added to the 
            reconstruction loss, intended to make inter-subject representations more diverse. (b) A lower mask ratio (0.5 / 0.5 / 0.35), intended to give the encoders more visible signal per training step.\n\n

            **Methods.** VICReg run (20 epochs, learning rate 1e-5, mask ratios 0.7 / 0.7 / 0.5) is compared with the pretrained run in Section 4.1. Lower-mask ratio run (MSE only, 20 epochs, learning rate 1e-5, mask ratio 0.5 / 0.5 / 0.35) is shown separately in Section 4.2. \n\n

            **Preliminary result.** VICReg reconstructs less well than the MSE run (mean ROI-wise R² 0.346 and 
            0.352, against 0.466 and 0.473 for pretrained). Its gender balanced accuracy (0.565) is slightly above than pretrained (0.557), and its OCD balanced accuracy (0.473) is close to pretrained (0.481); both differences are within CV variation. The lower mask ratio gives the best reconstruction (0.508 for HCP-YA and 0.520 for OCD), with gender balanced accuracy 0.565 and OCD balanced accuracy 0.465. No variant produces a clear classification gain over the FC benchmark.
            """),
        mo.md(f"""
            ### 4.1 VICReg vs MSE only\n
              
            **What VICReg is.** VICReg (Variance-Invariance-Covariance Regularization, Bardes et al., 2022) is a 
            regularizer for representation learning. It adds three terms to the training loss: an invariance term 
            that keeps the embeddings of two views of the same input close; a variance term that keeps the spread of 
            each latent dimension across the batch above a minimum, which prevents collapse to a constant; and a 
            covariance term that penalises correlations between latent dimensions, so each dimension carries 
            different information. Here the reconstruction loss is kept and the VICReg term (weight 0.01) is added to it.\n\n
            
            **Why diversify inter-subject differences.** A reconstruction objective is largely satisfied by structure 
            shared across subjects, such as the common spatial and temporal organisation of BOLD signals. Differences 
            between subjects, which is what a classifier needs to separate gender or diagnosis, can make up a small 
            part of the latent variance. The variance and covariance terms push the latents to spread across 
            subjects and to use more independent dimensions, so individual differences are represented explicitly.\n\n
              
            **Expected effect on gender classification.** If the extra spread is subject-specific, a linear probe has 
            more separable directions to work with, and the latents should move closer to the FC benchmark. The cost 
            is that the regularizer competes with reconstruction, so reconstruction quality is expected to drop.\n\n
            
            **Observed.** Reconstruction dropped as expected (mean ROI-wise R² 0.346 and 0.352, against 0.466 and 
            0.473 for pretrained). Gender balanced accuracy rose only marginally (0.565 vs 0.557), and OCD balanced accuracy 
            was unchanged (0.473 vs 0.481). Within CV variation, the diversification has not yet translated into 
            better gender classification.\n\n
            
            Note that learning rate differ can differ slightly between runs (1e-5 vs 5e-5) so the effects could be confounded by this difference but it is highly inlikely to change the main picture.
            """),
        mo.md(probe_md(("A", "C"))),
        mo.accordion({"Show loss curve": mo.vstack([
            mo.md(f"""**Training loss (VICReg).** Reconstruction and VICReg total loss per epoch. Look for both terms 
            decreasing. A reconstruction loss that plateaus early while the VICReg term keeps falling would mean the 
            regularizer dominates training."""),
            fig_c_curves(),
        ])}),
        mo.md(f"""### 4.2 Lower mask ratio (MSE only)\n
            
            **Why a lower mask ratio.** The encoders are trained as masked autoencoders: parts of the input (spatial, 
            temporal and context tokens) are hidden and must be reconstructed. The mask ratio sets how much is hidden. 
            Runs A and C hide 70% of spatial and temporal tokens; run B hides 50% / 50% / 35%. A high mask ratio makes 
            the task hard and pushes the encoder towards coarse, globally redundant structure. A lower ratio leaves more 
            visible context, makes reconstruction easier, and lets the encoder learn finer, local dependencies between 
            regions and time points.\n\n

            **Expected effect on latents.** We expect latents that encode more of the actual signal detail and fewer 
            averaged patterns, with better reconstruction. Observed: reconstruction is the best of all variants 
            (mean ROI-wise R² 0.508 for HCP-YA and 0.520 for OCD).\n\n

            **Expected use for gender classification.** Sex-related differences in functional connectivity, where 
            present, are expected to be subtle and distributed. Finer-grained latents could expose such patterns to a 
            probe that currently sees only coarse structure. The risk is that finer detail also carries scanner, motion 
            and noise components, which a classifier may use without any sex-related signal. Observed gender balanced 
            accuracy is 0.565 (vs 0.557 for pretrained), a change within noise, so the hypothesis is not yet supported.\n\n
            """),
        mo.md(probe_md(("B",))),
        mo.accordion({"Show loss curve": mo.vstack([
            mo.md(f"""**Training loss, lower mask ratio).** Training and validation reconstruction loss per epoch. Look 
            for the two curves converging. Because the masked task is easier, a lower loss floor than pretrained run is expected."""),
            fig_b_curves()
        ])}),
        mo.md(f"""### 4.3 Balanced accuracy summary\n

            **All latent variants against the FC benchmark.** Balanced accuracy per cohort for the FC feature sets, 
            the HCP-Aging latents and checkpoints A (pretrained), B (Lower mask ratio) and C (VICReg), with dotted chance line. Look for any latent bar approaching the FC bars; none does in this analysis."""),
        fig_bacc([(FC_ROI[0], fc_getter(FC_ROI[1])), (MODEL_LABEL["baseline"], latent_getter("baseline")),
                  (MODEL_LABEL["A"], latent_getter("A")), (MODEL_LABEL["C"], latent_getter("C")),
                  (MODEL_LABEL["B"], latent_getter("B"))], height=307),
    ])
    return (section_4,)


# ════════════════════════════════════════════════════════════════════════════
# Cell: Conclusion
# Purpose: "Conclusion" page: summary of the findings across Sections I–IV. Edit the markdown here.
# Depends on: mo.
# Provides: conclusion.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.md(f"""
            ## Conclusion\n

            **Reconstruction.** DAPT on HCP-YA strongly improves BOLD reconstruction on both cohorts. Mean 
            ROI-wise R² rises from 0.034 (HCP) and 0.075 (OCD) with the HCP-Aging weights (baseline) to 0.466 and 0.473 
            after pretraining (MSE only), and to 0.508 and 0.520 with lower mask ratios. The VICReg variant (C) reconstructs less well (0.346 and 0.352).\n\n

            **Classification from latents.** Balanced accuracy stays close to chance. For OCD condition 
            (97 subjects) the latent probes reach 0.46 to 0.48 against 0.50 chance, with 5-fold CV SD of 0.13 to 
            0.15, so no latent set is distinguishable from chance on this cohort. For HCP-YA gender the probes 
            reach 0.54 to 0.57 (CV SD about 0.02 to 0.03). DAPT changes this by at most about 0.03.\n\n

            **FC benchmark.** FC-only probes match or beat the latent probes on both tasks: ROI-level FC gives 
            0.61 for HCP-YA gender (499) and 0.59 for OCD condition, and RSN-level FC gives 0.58 and 0.67. 
            The latents do not yet add information beyond FC features.\n\n

            **Post-training.** Finetuning on HCP-YA gender (single train/val/test split) gives test accuracies 
            from 0.49 to 0.63. The best run (attention pooling, encoders updated) reaches 0.63. Differences of 
            this size are within split-to-split variation, and these results are not directly comparable with 
            the CV numbers above.\n\n

            **Overall.** DAPT improves reconstruction of the out-of-distribution OCD and HCP-YA data, but the 
            resulting latents do not improve OCD classification or gender decoding over FC features in these 
            experiments. With 97 OCD subjects, the classification comparisons cannot separate any model from 
            chance.
            """),
    ])
    return (conclusion,)


# ════════════════════════════════════════════════════════════════════════════
# Cell: Appendix filter controls
# Purpose: dropdown filters (pooling, protocol) for the all-probe table in the Appendix.
#          Kept in a separate cell because marimo does not allow reading a UI element's value
#          in the cell that created it.
# Depends on: mo, PROBES.
# Provides: pool_dd, proto_dd.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(PROBES, mo):
    pool_dd = mo.ui.dropdown(options=["all"] + sorted({r["pooling"] or "" for r in PROBES}), value="all",
                             label="Pooling")
    proto_dd = mo.ui.dropdown(options=["all"] + sorted({r["split_protocol"] or "not recorded" for r in PROBES}),
                              value="all", label="Protocol")
    return pool_dd, proto_dd


# ════════════════════════════════════════════════════════════════════════════
# Cell: Appendix
# Purpose: "Appendix": report notes and the expandable table of all probe results with
#          pooling/protocol filters (controls from the filter cell above).
# Depends on: mo, PROBES, f3, pool_dd, proto_dd.
# Provides: appendix.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(PROBES, f3, mo, pool_dd, proto_dd):
    def probe_rows_md(pool, proto):
        pool = "" if pool == "all" else pool
        proto = "" if proto == "all" else proto
        out = []
        for r in PROBES:
            p = r["pooling"] or ""
            q = r["split_protocol"] or "not recorded"
            if (pool and p != pool) or (proto and q != proto):
                continue
            kind = "FC" if r["probe"] == "logreg_subject_fc" else "latent"
            status = f"duplicate of {r['duplicate_of']}" if r.get("duplicate_of") else "canonical"
            out.append(f"| {kind} | {r['cohort']} | {r['ckpt']} | {p} | {q}"
                       f"{' LEAKY' if r['leaky'] else ''} | {r['target_scope'] or 'not recorded'} | "
                       f"{r['n_subjects_total']} | {f3(r['balanced_acc_mean'])} [{f3(r['balanced_acc_sd'])}] | {status} |")
        return "\n".join(["| kind | cohort | ckpt | pooling | split | target | n | bacc [sd] | status |",
                          "|---|---|---|---|---|---|---|---|---|"] + out)

    expandable = mo.vstack([mo.hstack([pool_dd, proto_dd]), mo.md(probe_rows_md(pool_dd.value, proto_dd.value))])

    appendix = mo.vstack([
        mo.md("## Appendix\n- Headline: target sample, subject folds, feature pooling, 5-fold CV, seed 42.\n"
              "- HCP 499 is the DAPT cohort (encoders saw these subjects unlabelled); OCD 97 is out-of-sample.\n"
              "- LEAKY (split_protocol=sample) rows appear only in the expandable table and Section III.\n"
              "- Duplicate result files are flagged and excluded from headline cells."),
        mo.md("### All probe results (expandable)"),
        expandable,
    ])
    return (appendix,)


# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# Cell: Page navigation state
# Purpose: holds the currently selected page, shared by the tabs and the "Next section" button.
#          allow_self_loops=True is required: the tabs and the button are created in the page-tabs
#          cell, which also reads this state, and marimo would otherwise skip that re-run.
# Provides: get_tab (current page name), set_tab (change page).
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(mo):
    get_tab, set_tab = mo.state("Introduction", allow_self_loops=True)
    return get_tab, set_tab


# ════════════════════════════════════════════════════════════════════════════
# Cell: Page tabs
# Purpose: assembles the page: Introduction, I–IV, Conclusion and Appendix as tabs, in page order,
#          followed by a "Next section" button that opens the following page.
# Depends on: mo, intro, section_1..4, conclusion, appendix, get_tab, set_tab.
# ════════════════════════════════════════════════════════════════════════════
@app.cell
def _(appendix, conclusion, get_tab, intro, mo, section_1, section_2, section_3, section_4, set_tab):
    pages = {
        "Introduction": intro,
        "I. Baseline analysis": section_1,
        "II. Pre-training": section_2,
        "III. Post-training": section_3,
        "IV. Extras": section_4,
        "Conclusion": conclusion,
        "Appendix": appendix,
    }
    current = get_tab()
    # lazy=True: each page is serialized only when it is opened. Without it, every navigation rebuilds the
    # tabs widget and re-sends all pages (including the Plotly figures), which is what made switching slow.
    sections = mo.ui.tabs(pages, value=current if current in pages else "Introduction", lazy=True,
                          on_change=set_tab)

    keys = list(pages)
    idx = keys.index(current) if current in keys else 0
    # Scroll to the top on every page change. innerHTML does not run <script> tags, so the scroll is
    # triggered by an image load error, which runs its onerror handler.
    scroll_top = mo.Html('<img src="x" alt="" style="display:none" onerror="window.scrollTo(0, 0)">')
    if idx < len(keys) - 1:
        next_key = keys[idx + 1]
        next_btn = mo.ui.button(label=f"Next section: {next_key} →", on_click=lambda _: set_tab(next_key))
        page = mo.vstack([sections, mo.hstack([next_btn], justify="end"), scroll_top])
    else:
        page = mo.vstack([sections, scroll_top])
    page
    return


if __name__ == "__main__":
    app.run()
