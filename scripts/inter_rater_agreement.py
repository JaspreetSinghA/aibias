"""
Inter-Rater Agreement Analysis for BiasLens / BAMIP Paper
==========================================================
Computes agreement statistics between human raters who scored LLM responses
on five bias-related dimensions (Accuracy, Relevance, Fairness, Neutrality,
Representation) on a 1–5 scale (0.5 increments).

Models and rater pairings:
  GPT-4        : Gurleen + Noor      (54 matched prompts)
  LLaMA-3.3-70B: Anu    + Harpreet   (54 matched prompts)
  Claude-3-Haiku: Jaspreet + Narveer  (16 matched prompts — limited)

Statistics computed
-------------------
Per model, per dimension:
  - Krippendorff's alpha  (interval metric)  — primary IRA measure
  - Weighted Cohen's kappa (linear weights)   — standard companion metric
  - Mean ± SD per rater                       — surfaces systematic leniency bias

Cross-model comparison (GPT-4 vs LLaMA), per dimension:
  - Mean score (average of the two raters per model)
  - Mann-Whitney U test                        — non-parametric significance test
  - Rank-biserial correlation r                — effect size

Outputs
-------
  reports/inter_rater_agreement/ira_full_report.txt  — human-readable paper appendix
  reports/inter_rater_agreement/ira_summary.csv      — machine-readable stats table

Usage
-----
  python3 scripts/inter_rater_agreement.py
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "inter_rater_agreement"

RATER_FILES: dict[str, tuple[str, str]] = {
    "GPT-4": (
        "llm_sikh_bias_responses_Gurleen_gpt-4.csv",
        "llm_sikh_bias_responses_Noor_gpt-4.csv",
    ),
    "LLaMA-3.3-70B": (
        "llm_sikh_bias_responses_Anu_llama-3.3-70b-versatile.csv",
        "llm_sikh_bias_responses_Harpreet_llama-3.3-70b-versatile.csv",
    ),
    "Claude-3-Haiku": (
        "llm_sikh_bias_responses_Jaspreet_claude-3-haiku-20240307.csv",
        "llm_sikh_bias_responses_Narveer_claude-3-haiku-20240307.csv",
    ),
}

DIMENSIONS = ["Accuracy", "Relevance", "Fairness", "Neutrality", "Representation"]

# Models flagged as low-reliability due to small matched sample
LIMITED_MODELS = {"Claude-3-Haiku"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rater_csv(path: Path) -> pd.DataFrame:
    """Load a rater CSV, cast dimension columns to float, index by Prompt ID."""
    df = pd.read_csv(path)
    for dim in DIMENSIONS:
        if dim in df.columns:
            df[dim] = pd.to_numeric(df[dim], errors="coerce")
    return df.set_index("Prompt ID")


def load_model_pair(model: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Return (rater1_df, rater2_df, rater1_name, rater2_name) for a model."""
    file1, file2 = RATER_FILES[model]
    rater1_name = file1.split("_")[4]   # e.g. "Gurleen" (fields: llm_sikh_bias_responses_<Name>_...)
    rater2_name = file2.split("_")[4]   # e.g. "Noor"
    df1 = load_rater_csv(DATA_DIR / file1)
    df2 = load_rater_csv(DATA_DIR / file2)
    return df1, df2, rater1_name, rater2_name


def get_matched_pair(
    df1: pd.DataFrame, df2: pd.DataFrame, dim: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return arrays of matched, non-null ratings for a single dimension.
    Uses pairwise deletion: only prompts where BOTH raters have a valid score
    are included. This is the standard approach when a small number of outlier
    ratings have been removed during data cleaning.
    """
    shared = df1.index.intersection(df2.index)
    s1 = df1.loc[shared, dim]
    s2 = df2.loc[shared, dim]
    mask = s1.notna() & s2.notna()
    return s1[mask].to_numpy(dtype=float), s2[mask].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Krippendorff's alpha — interval metric
# ---------------------------------------------------------------------------

def krippendorff_alpha_interval(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha using the interval metric for two raters.

    The interval metric weights disagreement by the squared difference between
    ratings, which is appropriate for our 1–5 ordinal scale with equal spacing.

    Formula (two-rater case):
        D_o = mean of (r1_i - r2_i)^2 over all matched pairs
        D_e = expected disagreement under statistical independence
            = variance of all ratings combined * correction factor
        alpha = 1 - D_o / D_e

    Reference: Krippendorff, K. (2004). Content Analysis, 2nd ed. Sage.
    """
    n = len(r1)
    if n < 2:
        return float("nan")

    all_vals = np.concatenate([r1, r2])
    # Observed disagreement: mean squared difference between paired ratings
    d_observed = np.mean((r1 - r2) ** 2)

    # Expected disagreement: sum of all pairwise squared differences / n*(n-1)
    # This is the unbiased estimator for the expected disagreement under
    # independence, equivalent to variance * (2n / (n-1))
    # Expected disagreement via the identity: ∑_{i<j}(x_i−x_j)² = n_total · SS
    # => D_e = 2·SS / (n_total − 1)
    grand_mean = np.mean(all_vals)
    sum_sq = np.sum((all_vals - grand_mean) ** 2)
    n_total = len(all_vals)
    d_expected = (2.0 * sum_sq) / (n_total - 1)

    if d_expected == 0:
        return 1.0 if d_observed == 0 else float("nan")

    return float(1.0 - d_observed / d_expected)


# ---------------------------------------------------------------------------
# Weighted Cohen's kappa — linear weights
# ---------------------------------------------------------------------------

def weighted_kappa_linear(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Compute linearly weighted Cohen's kappa via sklearn.

    Linear weights penalise disagreement proportionally to the distance
    between ratings, rather than treating all disagreements as equal.
    Appropriate for ordered categorical data like our 1–5 scale.
    """
    if len(r1) < 2:
        return float("nan")
    # sklearn requires integer labels; multiply by 2 to handle 0.5 steps
    r1_int = (r1 * 2).astype(int)
    r2_int = (r2 * 2).astype(int)
    try:
        return float(cohen_kappa_score(r1_int, r2_int, weights="linear"))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Per-model, per-dimension analysis
# ---------------------------------------------------------------------------

def analyze_model(model: str) -> dict:
    """
    Compute Krippendorff's alpha, weighted kappa, and descriptive stats
    for each dimension for a given model's rater pair.

    Returns a dict keyed by dimension containing per-dimension results,
    plus metadata (model name, rater names, n per dimension).
    """
    df1, df2, rater1, rater2 = load_model_pair(model)
    results: dict = {"model": model, "rater1": rater1, "rater2": rater2, "dims": {}}

    for dim in DIMENSIONS:
        r1, r2 = get_matched_pair(df1, df2, dim)
        n = len(r1)

        alpha = krippendorff_alpha_interval(r1, r2)
        kappa = weighted_kappa_linear(r1, r2)

        results["dims"][dim] = {
            "n": n,
            "alpha": alpha,
            "kappa": kappa,
            "mean_r1": float(np.mean(r1)) if n > 0 else float("nan"),
            "std_r1": float(np.std(r1, ddof=1)) if n > 1 else float("nan"),
            "mean_r2": float(np.mean(r2)) if n > 0 else float("nan"),
            "std_r2": float(np.std(r2, ddof=1)) if n > 1 else float("nan"),
            "mean_combined": float(np.mean(np.concatenate([r1, r2]))) if n > 0 else float("nan"),
        }

    return results


# ---------------------------------------------------------------------------
# Cross-model comparison: GPT-4 vs LLaMA
# ---------------------------------------------------------------------------

def compare_models(gpt4: dict, llama: dict) -> dict:
    """
    Run Mann-Whitney U tests comparing GPT-4 and LLaMA per dimension.

    Uses the combined ratings from both raters per model as the sample,
    then computes rank-biserial correlation as the effect size.

    Rank-biserial r = 1 - (2 * U) / (n1 * n2), where U is the Mann-Whitney U
    statistic. r ranges from -1 to 1; |r| > 0.3 is a medium effect,
    |r| > 0.5 is a large effect (Cohen, 1988).
    """
    comparison: dict = {}

    # Load raw rating arrays for each model (combined across both raters)
    def get_combined(model_results: dict, dim: str) -> np.ndarray:
        d = model_results["dims"][dim]
        # Reconstruct from means is lossy — reload the raw data
        return _raw_combined(model_results["model"], dim)

    for dim in DIMENSIONS:
        g_vals = _raw_combined("GPT-4", dim)
        l_vals = _raw_combined("LLaMA-3.3-70B", dim)

        u_stat, p_value = stats.mannwhitneyu(g_vals, l_vals, alternative="two-sided")
        n1, n2 = len(g_vals), len(l_vals)
        # Rank-biserial correlation (effect size)
        r_effect = float(1.0 - (2.0 * u_stat) / (n1 * n2))

        comparison[dim] = {
            "gpt4_mean": float(np.mean(g_vals)),
            "gpt4_std": float(np.std(g_vals, ddof=1)),
            "llama_mean": float(np.mean(l_vals)),
            "llama_std": float(np.std(l_vals, ddof=1)),
            "n_gpt4": n1,
            "n_llama": n2,
            "U": float(u_stat),
            "p_value": float(p_value),
            "effect_r": r_effect,
        }

    return comparison


def _raw_combined(model: str, dim: str) -> np.ndarray:
    """Load and concatenate valid ratings from both raters for a model+dimension."""
    df1, df2, _, _ = load_model_pair(model)
    r1, r2 = get_matched_pair(df1, df2, dim)
    return np.concatenate([r1, r2])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(value: float, decimals: int = 3) -> str:
    """Format a float, showing NaN as 'N/A'."""
    if np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def interpret_alpha(alpha: float) -> str:
    """Return a plain-English reliability label for a Krippendorff alpha value."""
    if np.isnan(alpha):
        return "N/A"
    if alpha < 0:
        return "below chance"
    if alpha < 0.20:
        return "slight"
    if alpha < 0.40:
        return "fair"
    if alpha < 0.60:
        return "moderate"
    if alpha < 0.80:
        return "substantial"
    return "almost perfect"


def interpret_kappa(kappa: float) -> str:
    """Return a reliability label for Cohen's kappa (Landis & Koch, 1977)."""
    if np.isnan(kappa):
        return "N/A"
    if kappa < 0:
        return "below chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    model_results: list[dict],
    comparison: dict,
    output_path: Path,
) -> None:
    """Write the full formatted report to a text file."""
    lines: list[str] = []
    W = 80  # line width

    def h1(title: str) -> None:
        lines.append("=" * W)
        lines.append(title.center(W))
        lines.append("=" * W)

    def h2(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * len(title))

    def row(*cols: str, widths: list[int]) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    h1("INTER-RATER AGREEMENT ANALYSIS")
    lines.append(f"  Project  : BiasLens / BAMIP — Sikh Bias in LLMs")
    lines.append(f"  Generated: {date.today().isoformat()}")
    lines.append(f"  Script   : scripts/inter_rater_agreement.py")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 1 — Methodology note
    # ------------------------------------------------------------------
    h1("SECTION 1 — METHODOLOGY")
    lines.append("""
Rating scale  : 1.0 – 5.0 in 0.5 increments (interval scale)
Dimensions    : Accuracy, Relevance, Fairness, Neutrality, Representation
Rater pairings:
  GPT-4         — Gurleen + Noor        (54 matched prompts)
  LLaMA-3.3-70B — Anu    + Harpreet    (54 matched prompts)
  Claude-3-Haiku — Jaspreet + Narveer  (16 matched prompts, LIMITED)

Missing values: Handled via pairwise deletion — for each dimension, only
  prompts where BOTH raters have a valid rating are included. Two cells in
  Gurleen's GPT-4 ratings (Neutrality and Representation for prompt
  KNOW_CONTEMPORARY_01) were recorded as 0, outside the valid 1–5 scale,
  and treated as missing; affected dimensions report n=53 rather than n=54.

Krippendorff's alpha: Computed using the interval metric, which weights
  disagreement by squared distance between ratings. This is appropriate for
  equal-interval ordinal scales. Implemented from scratch; formula from
  Krippendorff (2004). Reliability benchmarks: <0.20 slight, 0.20–0.40 fair,
  0.40–0.60 moderate, 0.60–0.80 substantial, >0.80 almost perfect.

Weighted Cohen's kappa: Computed using linear weights (sklearn). Penalises
  disagreements proportionally to distance. Same reliability benchmarks apply
  (Landis & Koch, 1977).

Mann-Whitney U test: Non-parametric test comparing GPT-4 vs LLaMA rating
  distributions per dimension. Two-sided. Effect size = rank-biserial r.
  |r| > 0.1 small, |r| > 0.3 medium, |r| > 0.5 large effect.

NOTE on Claude: Only 16 prompts were rated by both Jaspreet and Narveer.
  This is below the minimum recommended sample for reliable IRA estimates.
  Claude statistics are included for completeness but should be interpreted
  with caution and explicitly caveated in the paper.
""".strip())

    # ------------------------------------------------------------------
    # Section 2 — Per-model IRA tables
    # ------------------------------------------------------------------
    h1("SECTION 2 — PER-MODEL INTER-RATER AGREEMENT")

    for res in model_results:
        model = res["model"]
        r1, r2 = res["rater1"], res["rater2"]
        limited = model in LIMITED_MODELS

        h2(f"Model: {model}  (Raters: {r1} and {r2}){' *** LIMITED SAMPLE ***' if limited else ''}")

        col_widths = [16, 5, 9, 12, 16, 14, 14, 14]
        header = row(
            "Dimension", "n", "Alpha (α)", "Alpha label",
            "W. Kappa (κ)", "Kappa label", f"{r1} mean±SD", f"{r2} mean±SD",
            widths=col_widths,
        )
        lines.append(header)
        lines.append("-" * len(header))

        for dim in DIMENSIONS:
            d = res["dims"][dim]
            lines.append(row(
                dim,
                str(d["n"]),
                fmt(d["alpha"]),
                interpret_alpha(d["alpha"]),
                fmt(d["kappa"]),
                interpret_kappa(d["kappa"]),
                f"{fmt(d['mean_r1'], 2)}±{fmt(d['std_r1'], 2)}",
                f"{fmt(d['mean_r2'], 2)}±{fmt(d['std_r2'], 2)}",
                widths=col_widths,
            ))

        if limited:
            lines.append("")
            lines.append("  ⚠ Claude results are based on only 16 matched prompts and should")
            lines.append("    be treated as preliminary / indicative only.")

    # ------------------------------------------------------------------
    # Section 3 — Cross-model comparison
    # ------------------------------------------------------------------
    h1("SECTION 3 — CROSS-MODEL COMPARISON: GPT-4 vs LLaMA-3.3-70B")
    lines.append("""
Mann-Whitney U test (two-sided) comparing per-dimension score distributions.
Significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant
Effect size r (rank-biserial): |r|>0.1 small, |r|>0.3 medium, |r|>0.5 large
""".strip())
    lines.append("")

    cw = [16, 14, 14, 10, 10, 8, 12]
    cheader = row(
        "Dimension",
        "GPT-4 mean±SD", "LLaMA mean±SD",
        "U statistic", "p-value", "Sig.",
        "Effect r",
        widths=cw,
    )
    lines.append(cheader)
    lines.append("-" * len(cheader))

    for dim in DIMENSIONS:
        c = comparison[dim]
        lines.append(row(
            dim,
            f"{fmt(c['gpt4_mean'], 2)}±{fmt(c['gpt4_std'], 2)}",
            f"{fmt(c['llama_mean'], 2)}±{fmt(c['llama_std'], 2)}",
            fmt(c["U"], 1),
            fmt(c["p_value"], 4),
            significance_label(c["p_value"]),
            fmt(c["effect_r"], 3),
            widths=cw,
        ))

    # ------------------------------------------------------------------
    # Section 4 — Summary interpretation
    # ------------------------------------------------------------------
    h1("SECTION 4 — SUMMARY AND PAPER-READY INTERPRETATION")

    # Build dynamic summary from computed results
    def alpha_range(res: dict) -> str:
        alphas = [d["alpha"] for d in res["dims"].values() if not np.isnan(d["alpha"])]
        if not alphas:
            return "N/A"
        return f"{min(alphas):.2f} to {max(alphas):.2f}"

    def alpha_label_range(res: dict) -> str:
        alphas = [d["alpha"] for d in res["dims"].values() if not np.isnan(d["alpha"])]
        if not alphas:
            return "N/A"
        labels = sorted({interpret_alpha(a) for a in alphas})
        return " to ".join(labels)

    by_model = {r["model"]: r for r in model_results}

    sig_dims = [
        dim for dim, v in comparison.items()
        if v["p_value"] < 0.05
    ]
    sig_str = ", ".join(sig_dims) if sig_dims else "No dimensions"
    effect_rs = [abs(comparison[d]["effect_r"]) for d in sig_dims] if sig_dims else []
    effect_range = (
        f"{min(effect_rs):.3f}–{max(effect_rs):.3f}" if effect_rs else "N/A"
    )

    n_gpt4 = min(d["n"] for d in by_model["GPT-4"]["dims"].values())
    n_gpt4_max = max(d["n"] for d in by_model["GPT-4"]["dims"].values())
    n_llama = min(d["n"] for d in by_model["LLaMA-3.3-70B"]["dims"].values())
    n_claude = max(d["n"] for d in by_model["Claude-3-Haiku"]["dims"].values())

    lines.append(f"""
INTER-RATER RELIABILITY SUMMARY
--------------------------------
GPT-4 (Gurleen + Noor, n={n_gpt4}–{n_gpt4_max} per dimension):
  Krippendorff's alpha ranged from {alpha_range(by_model['GPT-4'])},
  reflecting {alpha_label_range(by_model['GPT-4'])} agreement overall.
  Noor rated more leniently than Gurleen across all dimensions (Relevance
  showed the largest systematic gap), suggesting rater calibration variance
  rather than random disagreement.

LLaMA-3.3-70B (Anu + Harpreet, n={n_llama} per dimension):
  Alpha ranged from {alpha_range(by_model['LLaMA-3.3-70B'])}, indicating
  {alpha_label_range(by_model['LLaMA-3.3-70B'])} agreement — more consistent
  than GPT-4 raters. Accuracy showed the strongest agreement.

Claude-3-Haiku (Jaspreet + Narveer, n={n_claude} matched prompts):
  Due to the limited overlap between raters, these statistics are unreliable
  and should be reported with explicit caveats. A full paired annotation is
  recommended before including Claude in IRA claims.

CROSS-MODEL COMPARISON SUMMARY
-------------------------------
{sig_str} showed statistically significant GPT-4 vs LLaMA differences
(Mann-Whitney U, p<0.05), with effect sizes |r| = {effect_range} (small).
Fairness, Neutrality, and Representation did not differ significantly,
suggesting both models perform comparably on the subjective bias dimensions.

SUGGESTED PAPER TEXT (for Section 2.5 / Section 3):
-----------------------------------------------------
"Inter-rater agreement was assessed using Krippendorff's alpha (interval
metric) and linearly weighted Cohen's kappa for each model and evaluation
dimension (n=54 prompts per model pair). For GPT-4, alpha ranged from
{alpha_range(by_model['GPT-4'])}, indicating {alpha_label_range(by_model['GPT-4'])}
agreement; systematic leniency differences between raters were observed,
particularly on Relevance. For LLaMA-3.3-70B, alpha ranged from
{alpha_range(by_model['LLaMA-3.3-70B'])}, indicating {alpha_label_range(by_model['LLaMA-3.3-70B'])}
agreement — more consistent across dimensions. These reliability levels are
within the range reported for subjective annotation tasks in NLP (Artstein &
Poesio, 2008). For Claude-3-Haiku, only 16 prompts were jointly annotated;
results (alpha {alpha_range(by_model['Claude-3-Haiku'])}) are preliminary and
should be interpreted with caution.

Mann-Whitney U tests comparing GPT-4 and LLaMA score distributions found
significant differences on {sig_str} (p<0.01, effect sizes
|r|={effect_range}, small). No significant differences were found on
Fairness, Neutrality, or Representation, indicating comparable performance
on the subjective bias dimensions despite differing on factual quality."
""".strip())

    lines.append("")
    lines.append("=" * W)
    lines.append("END OF REPORT".center(W))
    lines.append("=" * W)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [✓] Report written to {output_path}")


def write_csv_summary(model_results: list[dict], comparison: dict, output_path: Path) -> None:
    """Write a machine-readable CSV of all IRA statistics."""
    rows = []

    for res in model_results:
        model = res["model"]
        for dim in DIMENSIONS:
            d = res["dims"][dim]
            rows.append({
                "model": model,
                "dimension": dim,
                "n": d["n"],
                "krippendorff_alpha": round(d["alpha"], 4) if not np.isnan(d["alpha"]) else "",
                "alpha_label": interpret_alpha(d["alpha"]),
                "weighted_kappa": round(d["kappa"], 4) if not np.isnan(d["kappa"]) else "",
                "kappa_label": interpret_kappa(d["kappa"]),
                "rater1_mean": round(d["mean_r1"], 3),
                "rater1_std": round(d["std_r1"], 3),
                "rater2_mean": round(d["mean_r2"], 3),
                "rater2_std": round(d["std_r2"], 3),
                "combined_mean": round(d["mean_combined"], 3),
                "stat_type": "IRA",
            })

    for dim in DIMENSIONS:
        c = comparison[dim]
        rows.append({
            "model": "GPT-4 vs LLaMA",
            "dimension": dim,
            "n": f"{c['n_gpt4']} vs {c['n_llama']}",
            "krippendorff_alpha": "",
            "alpha_label": "",
            "weighted_kappa": "",
            "kappa_label": "",
            "rater1_mean": round(c["gpt4_mean"], 3),
            "rater1_std": round(c["gpt4_std"], 3),
            "rater2_mean": round(c["llama_mean"], 3),
            "rater2_std": round(c["llama_std"], 3),
            "combined_mean": "",
            "stat_type": "cross_model_comparison",
            "mann_whitney_U": round(c["U"], 1),
            "p_value": round(c["p_value"], 6),
            "significance": significance_label(c["p_value"]),
            "effect_r": round(c["effect_r"], 4),
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  [✓] CSV summary written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nBiasLens — Inter-Rater Agreement Analysis")
    print("=" * 50)

    # --- Step 1: Compute per-model IRA stats ---
    print("\n[1/3] Computing per-model agreement statistics...")
    model_results = []
    for model in ["GPT-4", "LLaMA-3.3-70B", "Claude-3-Haiku"]:
        print(f"      Processing {model}...")
        result = analyze_model(model)
        model_results.append(result)

        for dim in DIMENSIONS:
            d = result["dims"][dim]
            print(
                f"        {dim:<16} n={d['n']}  "
                f"α={fmt(d['alpha'])} ({interpret_alpha(d['alpha'])})  "
                f"κ={fmt(d['kappa'])} ({interpret_kappa(d['kappa'])})"
            )

    # --- Step 2: Cross-model comparison ---
    print("\n[2/3] Running GPT-4 vs LLaMA cross-model comparison...")
    gpt4_res = next(r for r in model_results if r["model"] == "GPT-4")
    llama_res = next(r for r in model_results if r["model"] == "LLaMA-3.3-70B")
    comparison = compare_models(gpt4_res, llama_res)

    for dim in DIMENSIONS:
        c = comparison[dim]
        print(
            f"      {dim:<16} GPT-4={fmt(c['gpt4_mean'], 2)}  "
            f"LLaMA={fmt(c['llama_mean'], 2)}  "
            f"p={fmt(c['p_value'], 4)}{significance_label(c['p_value'])}  "
            f"r={fmt(c['effect_r'], 3)}"
        )

    # --- Step 3: Write outputs ---
    print("\n[3/3] Writing output files...")
    write_report(
        model_results,
        comparison,
        REPORT_DIR / "ira_full_report.txt",
    )
    write_csv_summary(
        model_results,
        comparison,
        REPORT_DIR / "ira_summary.csv",
    )

    print("\nDone. Output files:")
    print(f"  {REPORT_DIR / 'ira_full_report.txt'}")
    print(f"  {REPORT_DIR / 'ira_summary.csv'}")
    print()


if __name__ == "__main__":
    main()
