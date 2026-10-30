"""Flask web app + REST API.  Run from the project root:  python app/app.py
Then open http://127.0.0.1:5000"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from flask import Flask, jsonify, request, render_template
from recommender_core import Recommender

app = Flask(__name__)
print("Training models... (takes a few seconds)")
rec = Recommender()
print("Ready.")


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/users")
def users():
    return jsonify(rec.users())


@app.get("/api/history/<int:user_id>")
def history(user_id):
    return jsonify(rec.history(user_id))


@app.get("/api/recommend/<int:user_id>")
def recommend(user_id):
    """Example: /api/recommend/42?model=svd&k=10&diverse=true&lambda=0.85"""
    try:
        t0 = time.perf_counter()
        out = rec.recommend(user_id,
                            model=request.args.get("model", "svd"),
                            k=int(request.args.get("k", 10)),
                            diverse=request.args.get("diverse", "false").lower() == "true",
                            lam=float(request.args.get("lambda", 0.85)))
        out["response_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return jsonify(out)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False, port=5000)
