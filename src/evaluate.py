import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_regression_metrics(y_true, y_pred):
    """
    Computes regression metrics for rating prediction.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    # NMAE: Normalized Mean Absolute Error (normalized by the rating range 0-5)
    nmae = mae / 5.0
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'NMAE': nmae
    }

def evaluate_ranking_for_user(actual_ratings, predicted_ratings, k=10, relevance_threshold=3.5):
    """
    Evaluates ranking metrics (Precision@K, Recall@K, NDCG@K, MRR) for a single user.
    actual_ratings: dict of {food_id: true_rating}
    predicted_ratings: dict of {food_id: pred_rating}
    """
    # Relevance ground truth
    relevant_items = {fid for fid, r in actual_ratings.items() if r >= relevance_threshold}
    
    if not relevant_items:
        return None  # No relevant items in test set for this user
        
    # Sort items by predicted rating
    sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    top_k_recommendations = [fid for fid, pred in sorted_predictions[:k]]
    
    # Precision@K: fraction of recommended items that are relevant
    hits = len(set(top_k_recommendations) & relevant_items)
    precision = hits / k
    
    # Recall@K: fraction of relevant items that are recommended
    recall = hits / len(relevant_items)
    
    # NDCG@K
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG@K
    for rank, fid in enumerate(top_k_recommendations):
        if fid in relevant_items:
            dcg += 1.0 / np.log2(rank + 2)
            
    # Calculate IDCG@K
    ideal_hits = min(len(relevant_items), k)
    for rank in range(ideal_hits):
        idcg += 1.0 / np.log2(rank + 2)
        
    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for rank, fid in enumerate(top_k_recommendations):
        if fid in relevant_items:
            mrr = 1.0 / (rank + 1)
            break
            
    return {
        'Precision@K': precision,
        'Recall@K': recall,
        'NDCG@K': ndcg,
        'MRR': mrr
    }

def calculate_ranking_metrics(test_df, predict_func, k=10, relevance_threshold=3.5):
    """
    Computes average ranking metrics across all users in the test set.
    test_df: test ratings DataFrame
    predict_func: function that takes (userId, foodId) and returns predicted rating
    """
    user_metrics = []
    
    # Group test ratings by user
    for uid, group in test_df.groupby('userId'):
        actual = dict(zip(group['foodId'], group['rating']))
        predicted = {fid: predict_func(uid, fid) for fid in group['foodId']}
        
        metrics = evaluate_ranking_for_user(actual, predicted, k=k, relevance_threshold=relevance_threshold)
        if metrics is not None:
            user_metrics.append(metrics)
            
    if not user_metrics:
        return {
            'Precision@K': 0.0,
            'Recall@K': 0.0,
            'NDCG@K': 0.0,
            'MRR': 0.0
        }
        
    # Average across users
    avg_metrics = {}
    for key in ['Precision@K', 'Recall@K', 'NDCG@K', 'MRR']:
        avg_metrics[key] = np.mean([m[key] for m in user_metrics])
        
    return avg_metrics
