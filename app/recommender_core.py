"""Shared logic for the web app: load data, train models, recommend (+ optional MMR diversity)."""
import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBaseline, BaselineOnly

DATA = "data/cleaned_dataset.csv"
SEED = 42


class Recommender:
    def __init__(self, path=DATA):
        self.df = pd.read_csv(path)
        movies = self.df.drop_duplicates("movie_title").set_index("movie_title")
        self.genre = movies["genre"].to_dict()
        self.lang = movies["language"].to_dict()
        self.year = movies["release_year"].to_dict()
        data = Dataset.load_from_df(self.df[["user_id", "movie_title", "rating"]], Reader(rating_scale=(1, 5)))
        self.trainset = data.build_full_trainset()               # app uses ALL data
        self.models = {
            "svd": SVD(n_factors=20, n_epochs=40, lr_all=0.007, reg_all=0.2, random_state=SEED),
            "item": KNNBaseline(k=40, min_k=3, sim_options={"name": "pearson_baseline", "user_based": False, "shrinkage": 100}, verbose=False),
            "user": KNNBaseline(k=40, min_k=3, sim_options={"name": "pearson_baseline", "user_based": True, "shrinkage": 100}, verbose=False),
            "baseline": BaselineOnly(verbose=False),
        }
        for m in self.models.values():
            m.fit(self.trainset)
        self.items = list(self.genre)

    # ---------- helpers ----------
    def users(self):
        return sorted(self.df["user_id"].unique().tolist())

    def history(self, user_id):
        h = self.df[self.df["user_id"] == user_id].sort_values("rating", ascending=False)
        return h[["movie_title", "genre", "language", "release_year", "rating"]].to_dict("records")

    def _sim(self, a, b):
        return 0.5 * (self.genre[a] == self.genre[b]) + 0.5 * (self.lang[a] == self.lang[b])

    def _mmr(self, cands, lam, k):
        chosen, pool = [], list(cands)
        while pool and len(chosen) < k:
            best = max(pool, key=lambda c: lam * (c[0] - 1) / 4
                       - (1 - lam) * max((self._sim(c[1], j) for _, j in chosen), default=0))
            chosen.append(best); pool.remove(best)
        return chosen

    # ---------- main API ----------
    def recommend(self, user_id, model="svd", k=10, diverse=False, lam=0.85, candidates=50):
        if model not in self.models:
            raise ValueError(f"model must be one of {list(self.models)}")
        seen = set(self.df.loc[self.df["user_id"] == user_id, "movie_title"])
        algo = self.models[model]
        scores = sorted(((algo.predict(user_id, i).est, i) for i in self.items if i not in seen), reverse=True)
        picked = self._mmr(scores[:candidates], lam, k) if diverse else scores[:k]
        recs = [{"movie_title": i, "genre": self.genre[i], "language": self.lang[i],
                 "release_year": int(self.year[i]), "predicted_rating": round(est, 2)} for est, i in picked]
        return {"user_id": user_id, "model": model, "diverse": diverse, "lambda": lam if diverse else None,
                "known_user": bool(seen), "languages_in_list": len({r["language"] for r in recs}),
                "genres_in_list": len({r["genre"] for r in recs}), "recommendations": recs}
