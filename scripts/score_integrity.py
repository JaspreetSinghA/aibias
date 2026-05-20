import pandas as pd

df = pd.read_csv("../data/ratings/combined.csv")
df["integrity"] = df[["neutrality","fairness","representation"]].mean(axis=1)
df["misrep_flag"] = (
    (df["accuracy"] < 3) |
    (df["relevancy"] < 3) |
    (df["integrity"] < 3)
)
df.to_csv("../data/processed/ratings_with_integrity.csv", index=False)
print("✅ Integrity scores -> ../data/processed/ratings_with_integrity.csv") 