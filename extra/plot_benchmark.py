import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitpdf.general_helpers import customise_matplotlib_format


customise_matplotlib_format()

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
    _rhat_max = item["r_hat"].max()

    _temp = {"nsamp": _nsamp, "mean": _mean, "std": _std, "rhat_max": _rhat_max}
    results.append(_temp)
    print(_temp)

df_results = pd.DataFrame.from_dict(results)
df_results = df_results.sort_values(by="nsamp")
df_results.index = np.arange(len(df_results.index))

print(df_results)
df_results.info()

params = {"dpi": 300}

# rhat max
fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(df_results["nsamp"], df_results["rhat_max"], marker="x", zorder=4)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Maximum Rhat")

fig.tight_layout()

fig.savefig(
    "benchmark_rhat_max.pdf",
    bbox_inches="tight",
    dpi=params["dpi"],
)

# parameter recovery accuracy
true_val = 0.2

fig = plt.figure()
ax = fig.add_subplot()

ax.errorbar(
    df_results["nsamp"], df_results["mean"], yerr=df_results["std"], fmt="x", zorder=4
)

ax.axhline(y=true_val, ls="dashed", lw=2, color="C1", zorder=3)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Fit value")

fig.tight_layout()

fig.savefig(
    "benchmark_accuracy.pdf",
    bbox_inches="tight",
    dpi=params["dpi"],
)

# relative uncertainty
fig = plt.figure()
ax = fig.add_subplot()

_rel_error = 100.0 * df_results["std"] / df_results["mean"]

ax.scatter(df_results["nsamp"], _rel_error, marker="x", zorder=4)

ax.axhline(y=20.0, ls="dashed", lw=2, color="C1", zorder=3)
ax.axhline(y=10.0, ls="dashed", lw=2, color="C2", zorder=3)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Relative uncertainty (%)")

fig.tight_layout()

fig.savefig(
    "benchmark_rel_uncertainty.pdf",
    bbox_inches="tight",
    dpi=params["dpi"],
)

# parameter delta
fig = plt.figure()
ax = fig.add_subplot()

data_lo = (df_results["mean"] - df_results["std"] - true_val) / df_results["std"]
data_hi = (df_results["mean"] + df_results["std"] - true_val) / df_results["std"]
data_mean = 0.5 * (data_lo + data_hi)

ax.fill_between(df_results["nsamp"], data_lo, data_hi, color="lightgray", zorder=3)

ax.scatter(df_results["nsamp"], data_mean, marker="x", zorder=5)

ax.axhline(y=0, ls="dashed", lw=2, color="C0", zorder=4)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Delta (std)")

fig.tight_layout()

fig.savefig(
    "benchmark_delta_std.pdf",
    bbox_inches="tight",
    dpi=params["dpi"],
)

# precision and spread
fig = plt.figure()
ax = fig.add_subplot()

ax.hist(df_results["mean"], bins="auto", histtype="step", lw=2, zorder=3)

ax.axvline(x=true_val, ls="dashed", lw=2, color="C1", zorder=5)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Fit value")

fig.tight_layout()

fig.savefig(
    "benchmark_histogram.pdf",
    bbox_inches="tight",
    dpi=params["dpi"],
)

plt.show()
