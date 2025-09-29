from __future__ import annotations

from pathlib import Path
import pandas as pd

from qte_thesis.config import BLD_data, BLD_figures


def task_build_os_table(
    depends_on=BLD_data / "merged_full_real_data.csv",
    produces=BLD_figures/"os_table.tex"
    ):

    df = pd.read_csv(depends_on)
    os_subgroup = df[(df["estimator"] == "DS") & (df["se"] == "sigma3")].copy()
    cols = {
        "tau": "Quantile",
        "alpha_hat": r"$\hat{\alpha}$",
        "sigma_hat": "SD",
        "lo": "CI lower",
        "hi": "CI upper",
        "support": "Support",
    }

    table_df = (
        os_subgroup[["tau", "alpha_hat", "sigma_hat", "lo", "hi", "support"]]
        .rename(columns=cols)
        .sort_values("Quantile")
        .reset_index(drop=True)
    )
    table_df[[r"$\hat{\alpha}$", "SD", "CI lower", "CI upper"]] = (
        table_df[[r"$\hat{\alpha}$", "SD", "CI lower", "CI upper"]].round(5)
    )

    table_df["Quantile"] = table_df["Quantile"].map(lambda x: f"{x:.2f}")

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        float_format=lambda x: f"{x:.5f}",
    )

    produces.parent.mkdir(parents=True, exist_ok=True)
    produces.write_text(latex_str, encoding="utf-8")


def task_build_ci_length_table(
        depends_on=BLD_data / "monte_carlo_merged_v2.csv",
        produces=BLD_figures/"ci_length_table.tex"
    ):
    mc = pd.read_csv(depends_on)

    mc["ci/se"] = mc["ci"] + "/" + mc["se"]

    grouped = mc.groupby(["ci/se", "n", "tau", "mu", "p", "estimator"], dropna=False)
    out = grouped["ci_length"].mean()

    df = out.unstack(["ci/se", "estimator"]).sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        ci_se_order = df.columns.get_level_values(0).unique().tolist()
        est_levels = df.columns.get_level_values(1).unique().tolist()
        est_order = [e for e in ["DS", "OS"] if e in est_levels] + [
            e for e in est_levels if e not in ["DS", "OS"]
        ]
        df = df.reindex(columns=pd.MultiIndex.from_product([ci_se_order, est_order]))

    idx = df.index.to_frame().copy()
    if "mu" in idx.columns:
        idx["mu"] = idx["mu"].apply(lambda x: "" if pd.isna(x) else f"{int(round(float(x)))}")
    if "tau" in idx.columns:
        idx["tau"] = idx["tau"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    df.index = pd.MultiIndex.from_frame(idx)
    df = df.round(4)
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([(c, "") for c in df.columns])
    df.columns = pd.MultiIndex.from_tuples(
        [(a, rf"\multicolumn{{1}}{{c}}{{{b}}}") for a, b in df.columns],
        names=["ci/se", ""],
    )
    col_format = "l" * df.index.nlevels + "S" * df.shape[1]
    latex = df.to_latex(
        index=True,
        multirow=True,
        na_rep="",
        float_format="%.4f",
        column_format=col_format,
        escape=False,
        index_names=True,
    )

    produces.parent.mkdir(parents=True, exist_ok=True)
    produces.write_text(latex, encoding="utf-8")