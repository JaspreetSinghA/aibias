import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from pathlib import Path

df = pd.read_csv("data/processed/ratings_with_integrity.csv")
out_dir = Path("reports/figures"); out_dir.mkdir(parents=True, exist_ok=True)

for scen in df["scenario"].unique():
    sub = df[df["scenario"] == scen]
    ctrl = sub["name"].unique()[0]  # first = control
    results = []
    for model in sub["model"].unique():
        ctrl_scores = sub[(sub["model"] == model) & (sub["name"] == ctrl)]["score"]
        for name in sub["name"].unique()[1:]:
            test_scores = sub[(sub["model"] == model) & (sub["name"] == name)]["score"]
            delta = test_scores.mean() - ctrl_scores.mean()
            t, p = ttest_rel(test_scores, ctrl_scores)
            results.append({"model": model, "name": name, "delta": delta, "p": p})
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"reports/tables/{scen}_stats.csv", index=False)

    plt.figure(figsize=(4,3))
    sns.barplot(data=res_df, x="model", y="delta", hue="name")
    plt.axhline(0, color="k", linewidth=.5)
    plt.title(f"{scen.title()} ΔScore (Sikh – Control)")
    plt.ylabel("Mean difference")
    plt.tight_layout()
    plt.savefig(out_dir / f"{scen}_delta.png", dpi=300)
    plt.close()
print("📊 Plots and stats saved in reports/")
