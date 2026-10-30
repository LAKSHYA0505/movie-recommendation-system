"""Step 3: reduce the filter bubble with diversity re-ranking (MMR).
For each user: take the model's top-50 candidates, then pick 10 one by one, balancing
  predicted rating  vs  being different (genre/language) from films already picked.
lam = 1.0 -> pure accuracy (original list),  lower lam -> more diverse.
Run from the project root:  python src/diversify.py"""
import itertools
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD, KNNBaseline
from surprise.model_selection import train_test_split

K, CANDIDATES, LIKED, SEED = 10, 50, 4.0, 42
LAMBDAS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7]

df = pd.read_csv("data/cleaned_dataset.csv")
meta = df.drop_duplicates("movie_title").set_index("movie_title")[["genre", "language"]]
GENRE, LANG = meta["genre"].to_dict(), meta["language"].to_dict()    # plain dicts = fast lookups
data = Dataset.load_from_df(df[["user_id", "movie_title", "rating"]], Reader(rating_scale=(1, 5)))
trainset, testset = train_test_split(data, test_size=0.2, random_state=SEED)

liked_in_test = defaultdict(set)                      # films each user liked but model never saw
for uid, iid, r in testset:
    if r >= LIKED:
        liked_in_test[uid].add(iid)

pop = defaultdict(int)
for _, i, _ in trainset.all_ratings():
    pop[trainset.to_raw_iid(i)] += 1
all_items = [trainset.to_raw_iid(i) for i in trainset.all_items()]


def similarity(a, b):
    """1 = same genre AND language, 0.5 = one of them matches, 0 = completely different."""
    return 0.5 * (GENRE[a] == GENRE[b]) + 0.5 * (LANG[a] == LANG[b])


def candidates(algo, uid, seen):
    scores = [(algo.predict(uid, i).est, i) for i in all_items if i not in seen]
    return sorted(scores, reverse=True)[:CANDIDATES]


def mmr(cands, lam, k=K):
    """Maximal Marginal Relevance re-ranking."""
    chosen, pool = [], list(cands)
    while pool and len(chosen) < k:
        def score(c):
            est, item = c
            rel = (est - 1) / 4                                  # rating 1-5 -> 0-1
            sim = max((similarity(item, j) for _, j in chosen), default=0)
            return lam * rel - (1 - lam) * sim
        best = max(pool, key=score)
        chosen.append(best); pool.remove(best)
    return chosen


def evaluate(lists):
    recommended = set(itertools.chain.from_iterable(i for l in lists.values() for _, i in l))
    hits, ild, langs, genres, nov, avg_pred = [], [], [], [], [], []
    for uid, l in lists.items():
        items = [i for _, i in l]
        if liked_in_test[uid]:
            hits.append(len(liked_in_test[uid] & set(items)) > 0)
        ild.append(np.mean([1 - similarity(a, b) for a, b in itertools.combinations(items, 2)]))
        langs.append(len({LANG[i] for i in items}))
        genres.append(len({GENRE[i] for i in items}))
        nov.append(np.mean([-np.log2((pop[i] + 1) / trainset.n_users) for i in items]))
        avg_pred.append(np.mean([e for e, _ in l]))
    return {"avg_predicted_rating": np.mean(avg_pred), f"hit_rate@{K}": np.mean(hits),
            "catalogue_coverage": len(recommended) / len(all_items), "intra_list_diversity": np.mean(ild),
            "languages_per_list": np.mean(langs), "genres_per_list": np.mean(genres), "novelty_bits": np.mean(nov)}


models = {"SVD": SVD(n_factors=20, n_epochs=40, lr_all=0.007, reg_all=0.2, random_state=SEED),
          "Item-based CF": KNNBaseline(k=40, min_k=3, sim_options={"name": "pearson_baseline", "user_based": False, "shrinkage": 100}, verbose=False)}
rows = []
for name, algo in models.items():
    algo.fit(trainset)
    cands = {}
    for u in trainset.all_users():
        uid = trainset.to_raw_uid(u)
        seen = {trainset.to_raw_iid(i) for i, _ in trainset.ur[u]}
        cands[uid] = candidates(algo, uid, seen)
    for lam in LAMBDAS:
        lists = {uid: mmr(c, lam) for uid, c in cands.items()}
        rows.append({"model": name, "lambda": lam, **evaluate(lists)})
        print(f"{name:14s} lambda={lam:.2f}  languages/list={rows[-1]['languages_per_list']:.2f}  "
              f"hit_rate={rows[-1][f'hit_rate@{K}']:.3f}  avg_pred={rows[-1]['avg_predicted_rating']:.2f}")

out = pd.DataFrame(rows).round(3)
out.to_csv("reports/diversity_results.csv", index=False)
print("\n", out.to_string(index=False))

# chart: accuracy vs diversity trade-off
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, g in out.groupby("model"):
    ax[0].plot(g["lambda"], g["languages_per_list"], marker="o", label=name)
    ax[1].plot(g["lambda"], g["avg_predicted_rating"], marker="o", label=name)
ax[0].set(title="Diversity: languages per top-10 list", xlabel="lambda (1 = no re-ranking)"); ax[0].invert_xaxis()
ax[1].set(title="Relevance: average predicted rating", xlabel="lambda (1 = no re-ranking)"); ax[1].invert_xaxis()
for a in ax: a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("reports/diversity_tradeoff.png", dpi=120)
print("\nSaved reports/diversity_results.csv and reports/diversity_tradeoff.png")
