"""Step 2: train and compare recommenders with the Surprise library.
Run from the project root:  python src/train_evaluate.py"""
from collections import defaultdict
import itertools, pickle
import numpy as np, pandas as pd
from surprise import Dataset, Reader, BaselineOnly, KNNBaseline, SVD, accuracy
from surprise.model_selection import train_test_split

K = 10            # length of each recommendation list
LIKED = 4.0       # a rating >= 4 counts as "liked"
SEED = 42

df = pd.read_csv("data/cleaned_dataset.csv")
meta = df.drop_duplicates("movie_title").set_index("movie_title")[["genre", "language"]]
data = Dataset.load_from_df(df[["user_id", "movie_title", "rating"]], Reader(rating_scale=(1, 5)))
trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)

models = {
    "Baseline (biases)": BaselineOnly(verbose=False),
    "User-based CF":     KNNBaseline(k=40, min_k=3, sim_options={"name": "pearson_baseline", "user_based": True, "shrinkage": 100}, verbose=False),
    "Item-based CF":     KNNBaseline(k=40, min_k=3, sim_options={"name": "pearson_baseline", "user_based": False, "shrinkage": 100}, verbose=False),
    "SVD":               SVD(n_factors=20, n_epochs=40, lr_all=0.007, reg_all=0.2, random_state=SEED),
}

# popularity of each film in training data -> used for novelty
pop = defaultdict(int)
for _, iid, _ in trainset.all_ratings():
    pop[trainset.to_raw_iid(iid)] += 1
n_users = trainset.n_users


def precision_recall_at_k(preds, k=K, threshold=LIKED):
    """Standard Surprise-style: rank each user's test films by predicted rating."""
    per_user = defaultdict(list)
    for p in preds:
        per_user[p.uid].append((p.est, p.r_ui))
    precs, recs = [], []
    for ratings in per_user.values():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true >= threshold for _, true in ratings)
        top = ratings[:k]
        n_rec = sum(est >= threshold for est, _ in top)
        n_hit = sum((est >= threshold) and (true >= threshold) for est, true in top)
        if n_rec: precs.append(n_hit / n_rec)
        if n_rel: recs.append(n_hit / n_rel)
    return np.mean(precs), np.mean(recs)


def top_k_for_all_users(algo, k=K):
    """Top-K unseen films for every user, from the whole catalogue."""
    all_items = [trainset.to_raw_iid(i) for i in trainset.all_items()]
    tops = {}
    for inner_u in trainset.all_users():
        uid = trainset.to_raw_uid(inner_u)
        seen = {trainset.to_raw_iid(i) for i, _ in trainset.ur[inner_u]}
        scores = [(algo.predict(uid, iid).est, iid) for iid in all_items if iid not in seen]
        tops[uid] = [iid for _, iid in sorted(scores, reverse=True)[:k]]
    return tops, all_items


def diversity_metrics(tops, all_items):
    recommended = set(itertools.chain.from_iterable(tops.values()))
    ild, nov, langs = [], [], []
    for items in tops.values():
        pairs = list(itertools.combinations(items, 2))
        ild.append(np.mean([(meta.loc[a, "genre"] != meta.loc[b, "genre"]) or
                            (meta.loc[a, "language"] != meta.loc[b, "language"]) for a, b in pairs]))
        nov.append(np.mean([-np.log2((pop[i] + 1) / n_users) for i in items]))
        langs.append(meta.loc[items, "language"].nunique())
    return {
        "catalogue_coverage": len(recommended) / len(all_items),   # share of films ever recommended
        "intra_list_diversity": np.mean(ild),                      # 0-1, higher = more varied lists
        "avg_languages_per_list": np.mean(langs),
        "novelty_bits": np.mean(nov),                              # higher = less mainstream picks
    }


results, best_name, best_rmse = [], None, 9
for name, algo in models.items():
    algo.fit(trainset)
    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)
    p, r = precision_recall_at_k(preds)
    tops, all_items = top_k_for_all_users(algo)
    row = {"model": name, "RMSE": rmse, "MAE": mae, f"Precision@{K}": p, f"Recall@{K}": r,
           **diversity_metrics(tops, all_items)}
    results.append(row)
    print(f"{name:18s} RMSE={rmse:.3f}  P@{K}={p:.3f}  R@{K}={r:.3f}")
    if rmse < best_rmse:
        best_name, best_rmse, best_algo = name, rmse, algo

out = pd.DataFrame(results).round(3)
out.to_csv("reports/model_results.csv", index=False)
print("\n", out.to_string(index=False))

best_algo.fit(data.build_full_trainset())          # retrain best model on all data
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({"name": best_name, "algo": best_algo}, f)
print(f"\nBest model: {best_name} -> saved to models/best_model.pkl")
