import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure imports work when running as script or module
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import load_and_preprocess_data, split_ratings_data
from models import (
    ContentBasedRecommender,
    MemoryCollaborativeFiltering,
    FunkSVDRecommender,
    NCFRecommender
)
from evaluate import calculate_regression_metrics, calculate_ranking_metrics

def main():
    print("=== SOCIAL MEDIA FOOD RECOMMENDATION SYSTEM TRAINING ===")
    
    # Define paths relative to project root
    foods_path = os.path.join(
        PROJECT_ROOT,
        "Clean Dataset-20260614T040023Z-3-001",
        "Clean Dataset",
        "foods.csv"
    )
    ratings_path = os.path.join(
        PROJECT_ROOT,
        "Clean Dataset-20260614T040023Z-3-001",
        "Clean Dataset",
        "ratings.csv"
    )
    figures_dir = os.path.join(PROJECT_ROOT, "report", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load and Preprocess Data
    print("\n[1/5] Loading and preprocessing datasets...")
    df_foods, df_ratings = load_and_preprocess_data(foods_path, ratings_path)
    print(f"Loaded {len(df_foods)} foods and {len(df_ratings)} ratings.")
    
    # Split data (90% train, 10% test)
    train_df, test_df = split_ratings_data(df_ratings, test_size=0.1, random_state=42)
    print(f"Splits - Train: {len(train_df)} ratings, Test: {len(test_df)} ratings.")
    
    # 2. Initialize Models
    print("\n[2/5] Initializing recommender models...")
    models = {
        "Content-Based": ContentBasedRecommender(),
        "User-CF (Cosine)": MemoryCollaborativeFiltering(kind='user', similarity='cosine', k=15),
        "User-CF (Pearson)": MemoryCollaborativeFiltering(kind='user', similarity='pearson', k=15),
        "Item-CF (Cosine)": MemoryCollaborativeFiltering(kind='item', similarity='cosine', k=15),
        "Item-CF (Pearson)": MemoryCollaborativeFiltering(kind='item', similarity='pearson', k=15),
        "Funk SVD (MF)": FunkSVDRecommender(n_factors=15, lr=0.005, reg=0.02, epochs=15),
        "Neural CF (NCF)": NCFRecommender(embedding_dim=16, layers=[32, 16], epochs=8, batch_size=256, lr=0.001)
    }
    
    # 3. Train Models
    print("\n[3/5] Training models...")
    
    # Fit CBF
    print("  Training Content-Based...")
    models["Content-Based"].fit(df_foods, train_df)
    
    # Fit CF Models
    for name in ["User-CF (Cosine)", "User-CF (Pearson)", "Item-CF (Cosine)", "Item-CF (Pearson)"]:
        print(f"  Training {name}...")
        models[name].fit(train_df)
        
    # Fit Funk SVD
    print("  Training Funk SVD (Matrix Factorization)...")
    models["Funk SVD (MF)"].fit(train_df)
    
    # Fit NCF
    print("  Training Neural CF (Deep Learning)...")
    models["Neural CF (NCF)"].fit(train_df)
    
    # 4. Evaluate Models
    print("\n[4/5] Evaluating models on test set...")
    regression_results = {}
    ranking_results = {}
    
    y_true = test_df['rating'].values
    
    test_users = test_df['userId'].astype(int).values
    test_foods = test_df['foodId'].astype(int).values

    for name, model in models.items():
        print(f"  Evaluating {name}...", flush=True)
        y_pred = (
            model.batch_predict(test_users, test_foods)
            if hasattr(model, 'batch_predict')
            else np.array([
                model.predict_rating(int(uid), int(fid))
                for uid, fid in zip(test_users, test_foods)
            ])
        )
        
        # Calculate regression metrics
        reg_metrics = calculate_regression_metrics(y_true, y_pred)
        regression_results[name] = reg_metrics
        
        # Calculate ranking metrics (Precision@10, Recall@10, NDCG@10, MRR)
        predict_func = lambda u, f: model.predict_rating(u, f)
        rank_metrics = calculate_ranking_metrics(test_df, predict_func, k=10, relevance_threshold=3.5)
        ranking_results[name] = rank_metrics
        
    # 5. Format & Print Results Tables
    print("\n[5/5] Compiling results...")
    
    df_reg = pd.DataFrame(regression_results).T
    df_rank = pd.DataFrame(ranking_results).T
    
    print("\n=== RATING PREDICTION METRICS (REGRESSION) ===")
    print(df_reg.to_string(formatters={
        'MSE': '{:,.4f}'.format,
        'RMSE': '{:,.4f}'.format,
        'MAE': '{:,.4f}'.format,
        'NMAE': '{:,.4f}'.format
    }))
    
    print("\n=== TOP-10 RECOMMENDATION METRICS (RANKING) ===")
    print(df_rank.to_string(formatters={
        'Precision@K': '{:.4%}'.format,
        'Recall@K': '{:.4%}'.format,
        'NDCG@K': '{:.4%}'.format,
        'MRR': '{:.4%}'.format
    }))
    
    # Save results to CSV files for report inclusion
    df_reg.to_csv(os.path.join(figures_dir, "regression_metrics.csv"))
    df_rank.to_csv(os.path.join(figures_dir, "ranking_metrics.csv"))
    
    # Create Visualizations
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Regression Metrics Comparison (RMSE & MAE)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_reg))
    width = 0.35
    
    plt.bar(x - width/2, df_reg['RMSE'], width, label='RMSE', color='#4A90E2')
    plt.bar(x + width/2, df_reg['MAE'], width, label='MAE', color='#F5A623')
    
    plt.xlabel('Recommendation Models')
    plt.ylabel('Error Value (Lower is Better)')
    plt.title('Rating Prediction Error Comparison (RMSE & MAE)')
    plt.xticks(x, df_reg.index, rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "error_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 2: Ranking Metrics Comparison (Precision@10 & NDCG@10)
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df_rank['Precision@K'], width, label='Precision@10', color='#50E3C2')
    plt.bar(x + width/2, df_rank['NDCG@K'], width, label='NDCG@10', color='#B8E986')
    
    plt.xlabel('Recommendation Models')
    plt.ylabel('Score (Higher is Better)')
    plt.title('Top-10 Recommendation Quality Comparison')
    plt.xticks(x, df_rank.index, rotation=30, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "ranking_comparison.png"), dpi=300)
    plt.close()
    
    # Plot 3: Sparsity / Rating Distribution (Pie Chart or Distribution Plot)
    plt.figure(figsize=(8, 5))
    rating_counts = df_ratings['rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis')
    plt.xlabel('Rating Scores')
    plt.ylabel('Frequency')
    plt.title('Overall Rating Distribution in ViFoodRec')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "rating_distribution.png"), dpi=300)
    plt.close()
    
    print("\nVisualizations saved successfully to report/figures/ directory.")
    print("Training process finished.")

if __name__ == "__main__":
    main()
