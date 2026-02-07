#!/usr/bin/env python3
import pandas as pd


COLUMNS = [
    "id",
    "dwa_type",
    "height",
    "n",
    "greedy_k",
    "greedy_ms",
    "dsatur_k",
    "dsatur_ms",
    "rlf_k",
    "rlf_ms",
    "sat_k",
    "sat_ms",
]

ALLOWED_TYPES = {
    "parser",
    "super",
    "terminal",
    "terminal_post_prune",
    "unknown",
}


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=COLUMNS,
        engine="python",
    )

    df = df[~df["id"].isin(["id", "sat_k", "dwa_type"])].copy()
    df = df[df["dwa_type"].isin(ALLOWED_TYPES)].copy()

    for col in COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def win_counts(df: pd.DataFrame):
    k_cols = ["greedy_k", "dsatur_k", "rlf_k"]
    row_min = df[k_cols].min(axis=1, skipna=True)
    min_counts = df[k_cols].eq(row_min, axis=0).sum(axis=1)

    wins = {}
    ties = {}
    for col in k_cols:
        wins[col] = int(((df[col] == row_min) & (min_counts == 1)).sum())
        ties[col] = int(((df[col] == row_min) & (min_counts > 1)).sum())

    return wins, ties


def win_counts_by_type(df: pd.DataFrame):
    results = {}
    for dwa_type, group in df.groupby("dwa_type"):
        wins, ties = win_counts(group)
        results[dwa_type] = {
            "count": int(len(group)),
            "wins": wins,
            "ties": ties,
        }
    return results


def average_colors_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("dwa_type")[["greedy_k", "dsatur_k", "rlf_k"]].mean()


def better_than_greedy(df: pd.DataFrame):
    dsatur_better = df[df["dsatur_k"] < df["greedy_k"]][
        ["id", "dwa_type", "height", "n", "greedy_k", "dsatur_k"]
    ]
    rlf_better = df[df["rlf_k"] < df["greedy_k"]][
        ["id", "dwa_type", "height", "n", "greedy_k", "rlf_k"]
    ]
    return dsatur_better, rlf_better


def max_improvement(df: pd.DataFrame):
    dsatur_improvement = df["greedy_k"] - df["dsatur_k"]
    rlf_improvement = df["greedy_k"] - df["rlf_k"]

    dsatur_best = dsatur_improvement.max(skipna=True)
    rlf_best = rlf_improvement.max(skipna=True)

    dsatur_row = None
    if pd.notna(dsatur_best):
        dsatur_row = df.loc[dsatur_improvement.idxmax()]

    rlf_row = None
    if pd.notna(rlf_best):
        rlf_row = df.loc[rlf_improvement.idxmax()]

    return (dsatur_best, dsatur_row), (rlf_best, rlf_row)


def main():
    df = load_results("benchmark_results.txt")

    print("# Overall wins (strict) and ties")
    wins, ties = win_counts(df)
    print(f"Total graphs: {len(df)}")
    for algo in ["greedy_k", "dsatur_k", "rlf_k"]:
        print(f"{algo}: wins={wins[algo]}, ties={ties[algo]}")

    print("\n# Wins and ties by DWA type")
    by_type = win_counts_by_type(df)
    for dwa_type in sorted(by_type.keys()):
        entry = by_type[dwa_type]
        wins = entry["wins"]
        ties = entry["ties"]
        print(
            f"{dwa_type}: count={entry['count']} | "
            f"greedy wins={wins['greedy_k']} ties={ties['greedy_k']} | "
            f"dsatur wins={wins['dsatur_k']} ties={ties['dsatur_k']} | "
            f"rlf wins={wins['rlf_k']} ties={ties['rlf_k']}"
        )

    print("\n# Average colors by DWA type")
    avg = average_colors_by_type(df)
    print(avg.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n# Cases where DSATUR beats greedy")
    dsatur_better, rlf_better = better_than_greedy(df)
    if dsatur_better.empty:
        print("(none)")
    else:
        print(dsatur_better.to_string(index=False))

    print("\n# Cases where RLF beats greedy")
    if rlf_better.empty:
        print("(none)")
    else:
        print(rlf_better.to_string(index=False))

    print("\n# Max improvement over greedy")
    (dsatur_best, dsatur_row), (rlf_best, rlf_row) = max_improvement(df)

    if pd.isna(dsatur_best):
        print("DSATUR: (no valid comparisons)")
    else:
        print(
            "DSATUR: "
            f"max_improvement={int(dsatur_best)} | "
            f"id={dsatur_row['id']} | "
            f"greedy_k={int(dsatur_row['greedy_k'])} | "
            f"dsatur_k={int(dsatur_row['dsatur_k'])}"
        )

    if pd.isna(rlf_best):
        print("RLF: (no valid comparisons)")
    else:
        print(
            "RLF: "
            f"max_improvement={int(rlf_best)} | "
            f"id={rlf_row['id']} | "
            f"greedy_k={int(rlf_row['greedy_k'])} | "
            f"rlf_k={int(rlf_row['rlf_k'])}"
        )


if __name__ == "__main__":
    main()
