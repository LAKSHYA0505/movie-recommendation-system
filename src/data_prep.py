"""Step 1 of the pipeline: clean dataset.csv -> data/cleaned_dataset.csv
Run from the project root:  python src/data_prep.py"""
import pandas as pd

RAW = "data/dataset.csv"
CLEAN = "data/cleaned_dataset.csv"
LABELS = "data/dataset_labeled.csv"


def fix_movie_info(df):
    """Give every film one genre/language/year (its most common value)."""
    for col in ["genre", "language", "release_year"]:
        df[col] = df.groupby("movie_title")[col].transform(lambda s: s.mode()[0])
    return df


def find_bot_users(df, gap_seconds=60, min_ratings=10):
    """Users who post most of their ratings within seconds of each other."""
    gaps = df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    fast_share = (gaps < gap_seconds).groupby(df["user_id"]).mean()
    counts = df["user_id"].value_counts()
    return fast_share[(fast_share > 0.5) & (counts[fast_share.index] >= min_ratings)].index.tolist()


def find_review_bombing(df, window="2h", min_ones=5):
    """Many 1-star ratings on the same film inside a short time window."""
    ones = df[df["rating"] == 1.0].set_index("timestamp").sort_index()
    flagged = []
    for title, g in ones.groupby("movie_title"):
        burst = g["rating"].rolling(window).count()
        if burst.max() >= min_ones:
            end = burst.idxmax()
            flagged += g[(g.index > end - pd.Timedelta(window)) & (g.index <= end)]["row"].tolist()
    return flagged


def clean(df):
    report = {"rows_before": len(df)}
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["row"] = df.index
    df = fix_movie_info(df)

    bots = find_bot_users(df)
    report["bot_users_removed"] = bots
    report["bot_rows_removed"] = int(df["user_id"].isin(bots).sum())
    df = df[~df["user_id"].isin(bots)]

    bomb_rows = find_review_bombing(df)
    report["review_bombing_rows_removed"] = len(bomb_rows)
    df = df[~df["row"].isin(bomb_rows)]

    before = len(df)
    df = df.drop_duplicates(["user_id", "movie_title"], keep="first")   # keep first rating per pair
    report["duplicate_rows_removed"] = before - len(df)
    report["rows_after"] = len(df)
    return df.drop(columns="row"), report


def check_against_labels(raw, cleaned):
    """How many of the planted anomalies did the cleaning catch?"""
    lab = pd.read_csv(LABELS, parse_dates=["timestamp"])
    key = ["user_id", "movie_title", "timestamp"]
    kept = lab.merge(cleaned[key], on=key, how="left", indicator=True)
    kept["removed"] = kept["_merge"] == "left_only"
    summary = kept.groupby("anomaly_type")["removed"].agg(["sum", "count"])
    summary.columns = ["removed", "total"]
    return summary


if __name__ == "__main__":
    raw = pd.read_csv(RAW, parse_dates=["timestamp"])
    cleaned, report = clean(raw.copy())
    cleaned.to_csv(CLEAN, index=False)
    for k, v in report.items():
        print(f"{k:30s} {v}")
    print("\nAnomalies caught (removed / total):")
    print(check_against_labels(raw, cleaned))
    print(f"\nSaved {CLEAN}")
