"""Lightweight training script for low-memory environments."""
import gc
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_preprocess_data, split_ratings_data
from models import (
    ContentBasedRecommender,
    MemoryCollaborativeFiltering,
    FunkSVDRecommender,
    NCFRecommender,
)
from evaluate import calculate_regression_metrics, calculate_ranking_metrics


def main():
    print("=== LIGHTWEIGHT TRAINING (low-memory mode) ===", flush=True)

    foods_path = os.path.join(
        PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "Clean Dataset", "foods.csv"
    )
    ratings_path = os.path.join(
        PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "Clean Dataset", "ratings.csv"
    )
    figures_dir = os.path.join(PROJECT_ROOT, "report", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("[1/5] Loading data...", flush=True)
    df_foods, df_ratings = load_and_preprocess_data(foods_path, ratings_path)
    train_df, test_df = split_ratings_data(df_ratings, test_size=0.1, random_state=42)
    print(f"  Foods: {len(df_foods)}, Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    models = {}
    configs = [
        ("Content-Based", lambda: ContentBasedRecommender()),
        ("User-CF (Cosine)", lambda: MemoryCollaborativeFiltering(kind="user", similarity="cosine", k=15)),
        ("User-CF (Pearson)", lambda: MemoryCollaborativeFiltering(kind="user", similarity="pearson", k=15)),
        ("Item-CF (Cosine)", lambda: MemoryCollaborativeFiltering(kind="item", similarity="cosine", k=15)),
        ("Item-CF (Pearson)", lambda: MemoryCollaborativeFiltering(kind="item", similarity="pearson", k=15)),
        ("Funk SVD (MF)", lambda: FunkSVDRecommender(n_factors=10, lr=0.01, reg=0.02, epochs=8)),
        ("Neural CF (NCF)", lambda: NCFRecommender(embedding_dim=8, layers=[16], epochs=20, batch_size=512, lr=0.001)),
    ]

    print("[2/5] Training models one-by-one...", flush=True)
    for name, factory in configs:
        print(f"  -> {name}", flush=True)
        model = factory()
        if name == "Content-Based":
            model.fit(df_foods, train_df)
        else:
            model.fit(train_df)
        models[name] = model
        gc.collect()

    print("[3/5] Evaluating...", flush=True)
    regression_results, ranking_results = {}, {}
    y_true = test_df["rating"].values
    test_users = test_df["userId"].astype(int).values
    test_foods = test_df["foodId"].astype(int).values

    for name, model in models.items():
        print(f"  -> {name}", flush=True)
        if hasattr(model, "batch_predict"):
            y_pred = model.batch_predict(test_users, test_foods)
        else:
            y_pred = np.array([model.predict_rating(int(u), int(f)) for u, f in zip(test_users, test_foods)])
        regression_results[name] = calculate_regression_metrics(y_true, y_pred)
        ranking_results[name] = calculate_ranking_metrics(
            test_df, lambda u, f, m=model: m.predict_rating(u, f), k=10, relevance_threshold=3.5
        )
        gc.collect()

    print("[4/5] Saving metrics...", flush=True)
    df_reg = pd.DataFrame(regression_results).T
    df_rank = pd.DataFrame(ranking_results).T
    df_reg.to_csv(os.path.join(figures_dir, "regression_metrics.csv"))
    df_rank.to_csv(os.path.join(figures_dir, "ranking_metrics.csv"))

    print("\n=== REGRESSION ===")
    print(df_reg.round(4).to_string())
    print("\n=== RANKING ===")
    print(df_rank.round(4).to_string())

    print("[5/5] Plotting...", flush=True)
    sns.set_theme(style="whitegrid")
    x = np.arange(len(df_reg))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, df_reg["RMSE"], width, label="RMSE", color="#4A90E2")
    plt.bar(x + width / 2, df_reg["MAE"], width, label="MAE", color="#F5A623")
    plt.xlabel("Recommendation Models")
    plt.ylabel("Error Value (Lower is Better)")
    plt.title("Rating Prediction Error Comparison (RMSE & MAE)")
    plt.xticks(x, df_reg.index, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "error_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, df_rank["Precision@K"], width, label="Precision@10", color="#50E3C2")
    plt.bar(x + width / 2, df_rank["NDCG@K"], width, label="NDCG@10", color="#B8E986")
    plt.xlabel("Recommendation Models")
    plt.ylabel("Score (Higher is Better)")
    plt.title("Top-10 Recommendation Quality Comparison")
    plt.xticks(x, df_rank.index, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ranking_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    rating_counts = df_ratings["rating"].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
    plt.xlabel("Rating Scores")
    plt.ylabel("Frequency")
    plt.title("Overall Rating Distribution in ViFoodRec")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rating_distribution.png"), dpi=200)
    plt.close()

    print("Done! Results saved to report/figures/", flush=True)


if __name__ == "__main__":
    main()
