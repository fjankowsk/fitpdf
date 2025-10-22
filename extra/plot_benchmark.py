import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


files = glob.glob("fit_result_*.csv")
files = sorted(files)
print(files)

dfs = []

for item in files:
    df = pd.read_csv(item, index_col=0)
    df["filename"] = item
    dfs.append(df)


results = []

for item in dfs:
    _nsamp = item["filename"].iat[0]
    _nsamp = _nsamp.rstrip(".csv")
    _nsamp = _nsamp.lstrip("fit_result_")
    _nsamp = int(_nsamp)
    _mean = item.at["w[0]", "mean"]
    _std = item.at["w[0]", "sd"]

    _temp = {"nsamp": _nsamp, "mean": _mean, "std": _std}
    results.append(_temp)
    print(_temp)

df_results = pd.DataFrame.from_dict(results)
df_results = df_results.sort_values(by="nsamp")
df_results.index = np.arange(len(df_results.index))

print(df_results)
df_results.info()

# parameter recovery accuracy
fig = plt.figure()
ax = fig.add_subplot()

ax.errorbar(
    df_results["nsamp"], df_results["mean"], yerr=df_results["std"], fmt="x", zorder=4
)

ax.axhline(y=0.2, ls="dashed", lw=2, color="C1", zorder=3)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Fit value")

fig.tight_layout()

# precision and spread
fig = plt.figure()
ax = fig.add_subplot()

ax.hist(df_results["mean"], bins="auto", histtype="step", lw=2)

ax.axhline(y=0.2, ls="dashed", lw=2, color="C1", zorder=3)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Fit value")

fig.tight_layout()

plt.show()
