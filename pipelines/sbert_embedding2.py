import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============================================================
# GIAI ĐOẠN 7B - HYBRID RECOMMENDATION: NCF + BERT EMBEDDING
#
# Input:
# - processed_data/train.csv
# - processed_data/validation.csv
# - processed_data/test.csv
# - processed_data/foods_processed.csv
# - processed_data/food_mapping.csv
# - models/ncf/*.keras
# - models/bert/food_embeddings_normalized.npy
# - models/bert/bert_config.json
#
# Output:
# - models/hybrid_bert/bert_user_profiles.npy
# - models/hybrid_bert/hybrid_bert_config.json
# - results/hybrid_bert_validation_metrics.csv
# - results/hybrid_bert_test_metrics.csv
# - results/hybrid_bert_alpha_tuning.png
# - results/ncf_vs_tfidf_vs_bert_hybrid.csv
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
DATA_DIR = "processed_data"
MODEL_DIR = "models"
NCF_MODEL_DIR = f"{MODEL_DIR}/ncf"
BERT_MODEL_DIR = f"{MODEL_DIR}/bert"
HYBRID_BERT_MODEL_DIR = f"{MODEL_DIR}/hybrid_bert"
RESULT_DIR = "results"

os.makedirs(HYBRID_BERT_MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RATING_MIN = 0.0
RATING_MAX = 5.0

RELEVANT_THRESHOLD = 4.0
TOP_LIKED_ITEMS = 20

TOP_K_LIST = [5, 10, 15, 20]

# alpha = 1.0: chỉ dùng NCF
# alpha = 0.0: chỉ dùng BERT Content-Based
ALPHA_LIST = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0]

# ------------------------------------------------------------
# 2. ĐỌC DỮ LIỆU
# ------------------------------------------------------------
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/validation.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

foods_df = pd.read_csv(f"{DATA_DIR}/foods_processed.csv")
food_mapping_df = pd.read_csv(f"{DATA_DIR}/food_mapping.csv")

train_val_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

print("=== KÍCH THƯỚC DỮ LIỆU ===")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)
print("Foods:", foods_df.shape)

# ------------------------------------------------------------
# 3. KIỂM TRA CỘT CẦN THIẾT
# ------------------------------------------------------------
required_rating_columns = [
    "user_encoded",
    "food_encoded",
    "rating"
]

for column in required_rating_columns:
    if column not in train_df.columns:
        raise ValueError(
            f"Không tìm thấy cột '{column}' trong train.csv."
        )

required_food_columns = [
    "food_id"
]

for column in required_food_columns:
    if column not in foods_df.columns:
        raise ValueError(
            f"Không tìm thấy cột '{column}' trong foods_processed.csv."
        )

required_mapping_columns = [
    "food_id",
    "food_encoded"
]

for column in required_mapping_columns:
    if column not in food_mapping_df.columns:
        raise ValueError(
            f"Không tìm thấy cột '{column}' trong food_mapping.csv."
        )

# ------------------------------------------------------------
# 4. GHÉP FOOD ENCODED VÀO DATASET MÓN ĂN
# ------------------------------------------------------------
foods_df = foods_df.merge(
    food_mapping_df[
        ["food_id", "food_encoded"]
    ],
    on="food_id",
    how="inner"
)

foods_df = foods_df.sort_values(
    "food_encoded"
).reset_index(drop=True)

num_items = foods_df["food_encoded"].nunique()

num_users = (
    max(
        train_df["user_encoded"].max(),
        val_df["user_encoded"].max(),
        test_df["user_encoded"].max()
    ) + 1
)

expected_item_ids = set(range(num_items))
actual_item_ids = set(
    foods_df["food_encoded"].unique()
)

if expected_item_ids != actual_item_ids:
    raise ValueError(
        "food_encoded không liên tục từ 0 đến num_items - 1."
    )

print("\n=== THÔNG TIN ENCODE ===")
print("Số user:", num_users)
print("Số món:", num_items)

# ------------------------------------------------------------
# 5. LOAD BERT EMBEDDING
# ------------------------------------------------------------
bert_embedding_path = (
    f"{BERT_MODEL_DIR}/food_embeddings_normalized.npy"
)

bert_config_path = (
    f"{BERT_MODEL_DIR}/bert_config.json"
)

if not os.path.exists(bert_embedding_path):
    raise FileNotFoundError(
        "Không tìm thấy food_embeddings_normalized.npy. "
        "Hãy chạy Giai đoạn 7A trước."
    )

food_embeddings = np.load(
    bert_embedding_path
).astype(np.float32)

with open(
    bert_config_path,
    "r",
    encoding="utf-8"
) as file:
    bert_config = json.load(file)

print("\n=== BERT EMBEDDING ===")
print("Embedding shape:", food_embeddings.shape)
print("Embedding model:", bert_config["model_name"])

if food_embeddings.shape[0] != num_items:
    raise ValueError(
        "Số embedding không khớp số món ăn. "
        "Kiểm tra food_mapping.csv và embedding."
    )

embedding_dim = food_embeddings.shape[1]

# ------------------------------------------------------------
# 6. CHỌN NCF MODEL TỐT NHẤT
# ------------------------------------------------------------
def find_best_ncf_model():
    """
    Chọn NCF có NDCG@10 tốt nhất từ Giai đoạn 4.
    """
    metrics_path = f"{RESULT_DIR}/ncf_top_n_metrics.csv"

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)

        ndcg_10_df = metrics_df[
            metrics_df["K"] == 10
        ].copy()

        if len(ndcg_10_df) > 0:
            best_model_name = (
                ndcg_10_df
                .sort_values(
                    "NDCG@K",
                    ascending=False
                )
                .iloc[0]["Model"]
            )

            return best_model_name

    return "NeuMF"


best_ncf_name = find_best_ncf_model()

ncf_model_path = (
    f"{NCF_MODEL_DIR}/{best_ncf_name}_final.keras"
)

if not os.path.exists(ncf_model_path):
    raise FileNotFoundError(
        f"Không tìm thấy NCF model: {ncf_model_path}"
    )

ncf_model = tf.keras.models.load_model(
    ncf_model_path
)

print("\n=== NCF MODEL ===")
print("Model được chọn:", best_ncf_name)

# ------------------------------------------------------------
# 7. TẠO BERT USER PROFILE
# ------------------------------------------------------------
def build_bert_user_profiles(
    rating_data,
    num_users,
    food_embeddings,
    relevant_threshold=4.0,
    top_liked_items=20
):
    """
    User profile = weighted average embedding của các món
    user đánh giá cao.

    Weight chính là rating.
    """
    embedding_dim = food_embeddings.shape[1]

    user_profiles = np.zeros(
        (num_users, embedding_dim),
        dtype=np.float32
    )

    for user_id in range(num_users):
        user_ratings = rating_data[
            rating_data["user_encoded"] == user_id
        ].copy()

        liked_items = user_ratings[
            user_ratings["rating"] >= relevant_threshold
        ].sort_values(
            "rating",
            ascending=False
        ).head(top_liked_items)

        if len(liked_items) == 0:
            continue

        item_ids = liked_items[
            "food_encoded"
        ].values.astype(int)

        weights = liked_items[
            "rating"
        ].values.astype(np.float32)

        profile = np.average(
            food_embeddings[item_ids],
            axis=0,
            weights=weights
        )

        profile_norm = np.linalg.norm(profile)

        if profile_norm > 0:
            profile = profile / profile_norm

        user_profiles[user_id] = profile

    return user_profiles


print("\nĐang tạo BERT User Profile từ train...")

bert_profiles_train = build_bert_user_profiles(
    rating_data=train_df,
    num_users=num_users,
    food_embeddings=food_embeddings,
    relevant_threshold=RELEVANT_THRESHOLD,
    top_liked_items=TOP_LIKED_ITEMS
)

print("BERT User Profile train shape:", bert_profiles_train.shape)

# ------------------------------------------------------------
# 8. HÀM TÍNH NCF SCORE
# ------------------------------------------------------------
def predict_ncf_scores(
    model,
    user_id,
    candidate_items,
    batch_size=1024
):
    candidate_items = np.asarray(
        candidate_items,
        dtype=np.int32
    )

    user_array = np.full(
        len(candidate_items),
        user_id,
        dtype=np.int32
    )

    predictions = model.predict(
        {
            "user_input": user_array,
            "item_input": candidate_items
        },
        batch_size=batch_size,
        verbose=0
    ).flatten()

    return np.clip(
        predictions,
        RATING_MIN,
        RATING_MAX
    )


# ------------------------------------------------------------
# 9. HÀM TÍNH BERT CONTENT SCORE
# ------------------------------------------------------------
def get_bert_content_scores(
    user_id,
    candidate_items,
    user_profiles,
    food_embeddings
):
    """
    Vì embedding và profile đã L2-normalize:
    cosine similarity = dot product.
    """
    profile = user_profiles[user_id]

    profile_norm = np.linalg.norm(profile)

    if profile_norm == 0:
        return np.zeros(len(candidate_items))

    candidate_embeddings = food_embeddings[
        candidate_items
    ]

    return np.dot(
        candidate_embeddings,
        profile
    )


# ------------------------------------------------------------
# 10. NORMALIZE SCORE
# ------------------------------------------------------------
def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return scores

    min_score = scores.min()
    max_score = scores.max()

    if max_score - min_score == 0:
        return np.zeros_like(scores)

    return (
        (scores - min_score) /
        (max_score - min_score)
    )


# ------------------------------------------------------------
# 11. HYBRID NCF + BERT SCORE
# ------------------------------------------------------------
def hybrid_bert_scores(
    ncf_model,
    user_id,
    candidate_items,
    user_profiles,
    food_embeddings,
    alpha=0.7
):
    """
    Final Score =
    alpha * normalized_NCF +
    (1 - alpha) * normalized_BERT
    """
    ncf_scores = predict_ncf_scores(
        model=ncf_model,
        user_id=user_id,
        candidate_items=candidate_items
    )

    bert_scores = get_bert_content_scores(
        user_id=user_id,
        candidate_items=candidate_items,
        user_profiles=user_profiles,
        food_embeddings=food_embeddings
    )

    ncf_normalized = min_max_normalize(
        ncf_scores
    )

    bert_normalized = min_max_normalize(
        bert_scores
    )

    final_scores = (
        alpha * ncf_normalized +
        (1 - alpha) * bert_normalized
    )

    return final_scores, ncf_scores, bert_scores


# ------------------------------------------------------------
# 12. TOP-N RECOMMENDATION
# ------------------------------------------------------------
def recommend_hybrid_bert_top_n(
    ncf_model,
    user_id,
    seen_items,
    num_items,
    user_profiles,
    food_embeddings,
    alpha=0.7,
    top_n=10
):
    candidate_items = np.array(
        [
            item_id
            for item_id in range(num_items)
            if item_id not in seen_items
        ],
        dtype=np.int32
    )

    if len(candidate_items) == 0:
        return []

    final_scores, _, _ = hybrid_bert_scores(
        ncf_model=ncf_model,
        user_id=user_id,
        candidate_items=candidate_items,
        user_profiles=user_profiles,
        food_embeddings=food_embeddings,
        alpha=alpha
    )

    top_indices = np.argsort(
        final_scores
    )[::-1][:top_n]

    return candidate_items[top_indices].tolist()


# ------------------------------------------------------------
# 13. TOP-N METRICS
# ------------------------------------------------------------
def precision_recall_f1_at_k(
    recommended_items,
    relevant_items,
    k
):
    recommended_k = recommended_items[:k]

    if len(recommended_k) == 0 or len(relevant_items) == 0:
        return 0.0, 0.0, 0.0

    hits = len(
        set(recommended_k) &
        set(relevant_items)
    )

    precision = hits / k
    recall = hits / len(relevant_items)

    f1 = 0.0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    return precision, recall, f1


def ndcg_at_k(
    recommended_items,
    relevant_items,
    k
):
    dcg = 0.0

    for index, item_id in enumerate(
        recommended_items[:k]
    ):
        if item_id in relevant_items:
            dcg += 1 / np.log2(index + 2)

    ideal_hits = min(
        len(relevant_items),
        k
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1 / np.log2(index + 2)
        for index in range(ideal_hits)
    )

    return dcg / idcg


def mrr_at_k(
    recommended_items,
    relevant_items,
    k
):
    for index, item_id in enumerate(
        recommended_items[:k]
    ):
        if item_id in relevant_items:
            return 1 / (index + 1)

    return 0.0


# ------------------------------------------------------------
# 14. ĐÁNH GIÁ HYBRID BERT
# ------------------------------------------------------------
def evaluate_hybrid_bert_top_n(
    ncf_model,
    train_data,
    eval_data,
    num_items,
    user_profiles,
    food_embeddings,
    alpha,
    k_list,
    threshold=4.0
):
    relevant_items_by_user = (
        eval_data[
            eval_data["rating"] >= threshold
        ]
        .groupby("user_encoded")["food_encoded"]
        .apply(set)
        .to_dict()
    )

    seen_items_by_user = (
        train_data
        .groupby("user_encoded")["food_encoded"]
        .apply(set)
        .to_dict()
    )

    rows = []

    for k in k_list:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []
        mrr_scores = []

        for user_id, relevant_items in relevant_items_by_user.items():
            recommended_items = recommend_hybrid_bert_top_n(
                ncf_model=ncf_model,
                user_id=user_id,
                seen_items=seen_items_by_user.get(
                    user_id,
                    set()
                ),
                num_items=num_items,
                user_profiles=user_profiles,
                food_embeddings=food_embeddings,
                alpha=alpha,
                top_n=k
            )

            precision, recall, f1 = (
                precision_recall_f1_at_k(
                    recommended_items,
                    relevant_items,
                    k
                )
            )

            ndcg = ndcg_at_k(
                recommended_items,
                relevant_items,
                k
            )

            mrr = mrr_at_k(
                recommended_items,
                relevant_items,
                k
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)

        rows.append({
            "Model": f"Hybrid_BERT_{best_ncf_name}",
            "Alpha_NCF": alpha,
            "K": k,
            "Precision@K": np.mean(precision_scores),
            "Recall@K": np.mean(recall_scores),
            "F1@K": np.mean(f1_scores),
            "NDCG@K": np.mean(ndcg_scores),
            "MRR@K": np.mean(mrr_scores)
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 15. TUNING ALPHA TRÊN VALIDATION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("TUNING ALPHA: NCF + BERT")
print("=" * 60)

validation_results = []

for alpha in ALPHA_LIST:
    print(f"Đang đánh giá alpha = {alpha}")

    result_df = evaluate_hybrid_bert_top_n(
        ncf_model=ncf_model,
        train_data=train_df,
        eval_data=val_df,
        num_items=num_items,
        user_profiles=bert_profiles_train,
        food_embeddings=food_embeddings,
        alpha=alpha,
        k_list=TOP_K_LIST,
        threshold=RELEVANT_THRESHOLD
    )

    validation_results.append(result_df)

hybrid_bert_validation_df = pd.concat(
    validation_results,
    ignore_index=True
)

hybrid_bert_validation_df.to_csv(
    f"{RESULT_DIR}/hybrid_bert_validation_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

best_alpha_row = (
    hybrid_bert_validation_df[
        hybrid_bert_validation_df["K"] == 10
    ]
    .sort_values(
        "NDCG@K",
        ascending=False
    )
    .iloc[0]
)

best_alpha = float(
    best_alpha_row["Alpha_NCF"]
)

print("\n=== ALPHA TỐT NHẤT ===")
print("Best alpha:", best_alpha)
print(
    "Best Validation NDCG@10:",
    round(best_alpha_row["NDCG@K"], 4)
)

# ------------------------------------------------------------
# 16. VẼ BIỂU ĐỒ TUNING ALPHA
# ------------------------------------------------------------
alpha_ndcg_10 = (
    hybrid_bert_validation_df[
        hybrid_bert_validation_df["K"] == 10
    ]
    .sort_values("Alpha_NCF")
)

plt.figure(figsize=(10, 6))

plt.plot(
    alpha_ndcg_10["Alpha_NCF"],
    alpha_ndcg_10["NDCG@K"],
    marker="o"
)

plt.title("Tuning Alpha cho Hybrid NCF + BERT")
plt.xlabel("Alpha cho NCF")
plt.ylabel("NDCG@10")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/hybrid_bert_alpha_tuning.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 17. TẠO PROFILE TỪ TRAIN + VALIDATION
# ------------------------------------------------------------
print("\nĐang tạo BERT User Profile từ train + validation...")

bert_profiles_train_val = build_bert_user_profiles(
    rating_data=train_val_df,
    num_users=num_users,
    food_embeddings=food_embeddings,
    relevant_threshold=RELEVANT_THRESHOLD,
    top_liked_items=TOP_LIKED_ITEMS
)

np.save(
    f"{HYBRID_BERT_MODEL_DIR}/bert_user_profiles.npy",
    bert_profiles_train_val
)

# ------------------------------------------------------------
# 18. ĐÁNH GIÁ TRÊN TEST
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("ĐÁNH GIÁ HYBRID NCF + BERT TRÊN TEST")
print("=" * 60)

hybrid_bert_test_df = evaluate_hybrid_bert_top_n(
    ncf_model=ncf_model,
    train_data=train_val_df,
    eval_data=test_df,
    num_items=num_items,
    user_profiles=bert_profiles_train_val,
    food_embeddings=food_embeddings,
    alpha=best_alpha,
    k_list=TOP_K_LIST,
    threshold=RELEVANT_THRESHOLD
)

hybrid_bert_test_df.to_csv(
    f"{RESULT_DIR}/hybrid_bert_test_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(hybrid_bert_test_df)

# ------------------------------------------------------------
# 19. ĐÁNH GIÁ RATING PREDICTION
# ------------------------------------------------------------
def evaluate_hybrid_bert_rating_prediction(
    ncf_model,
    eval_data,
    user_profiles,
    food_embeddings,
    alpha
):
    y_true = []
    y_pred = []

    for user_id, user_group in eval_data.groupby(
        "user_encoded"
    ):
        candidate_items = user_group[
            "food_encoded"
        ].values.astype(int)

        final_scores, _, _ = hybrid_bert_scores(
            ncf_model=ncf_model,
            user_id=user_id,
            candidate_items=candidate_items,
            user_profiles=user_profiles,
            food_embeddings=food_embeddings,
            alpha=alpha
        )

        predicted_ratings = (
            final_scores *
            (RATING_MAX - RATING_MIN) +
            RATING_MIN
        )

        y_true.extend(
            user_group["rating"].values
        )

        y_pred.extend(
            predicted_ratings
        )

    y_true = np.array(y_true)
    y_pred = np.clip(
        np.array(y_pred),
        RATING_MIN,
        RATING_MAX
    )

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    nmae = mae / (
        RATING_MAX - RATING_MIN
    )

    return {
        "Model": f"Hybrid_BERT_{best_ncf_name}",
        "Alpha_NCF": alpha,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "NMAE": nmae
    }


hybrid_bert_rating_result = (
    evaluate_hybrid_bert_rating_prediction(
        ncf_model=ncf_model,
        eval_data=test_df,
        user_profiles=bert_profiles_train_val,
        food_embeddings=food_embeddings,
        alpha=best_alpha
    )
)

hybrid_bert_rating_df = pd.DataFrame(
    [hybrid_bert_rating_result]
)

hybrid_bert_rating_df.to_csv(
    f"{RESULT_DIR}/hybrid_bert_rating_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== HYBRID BERT RATING METRICS ===")
print(hybrid_bert_rating_df)

# ------------------------------------------------------------
# 20. SO SÁNH NCF - TFIDF HYBRID - BERT HYBRID
# ------------------------------------------------------------
comparison_frames = []

ncf_metrics_path = (
    f"{RESULT_DIR}/ncf_top_n_metrics.csv"
)

tfidf_hybrid_metrics_path = (
    f"{RESULT_DIR}/hybrid_test_metrics.csv"
)

if os.path.exists(ncf_metrics_path):
    ncf_df = pd.read_csv(ncf_metrics_path)

    ncf_best_df = ncf_df[
        ncf_df["Model"] == best_ncf_name
    ].copy()

    comparison_frames.append(ncf_best_df)

if os.path.exists(tfidf_hybrid_metrics_path):
    tfidf_hybrid_df = pd.read_csv(
        tfidf_hybrid_metrics_path
    ).copy()

    tfidf_hybrid_df["Model"] = (
        f"Hybrid_TFIDF_{best_ncf_name}"
    )

    comparison_frames.append(tfidf_hybrid_df)

comparison_frames.append(
    hybrid_bert_test_df
)

comparison_df = pd.concat(
    comparison_frames,
    ignore_index=True
)

comparison_df.to_csv(
    f"{RESULT_DIR}/ncf_vs_tfidf_vs_bert_hybrid.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== SO SÁNH CÁC MÔ HÌNH ===")
print(comparison_df)

# Biểu đồ NDCG@K
plt.figure(figsize=(10, 6))

for model_name in comparison_df["Model"].unique():
    model_data = comparison_df[
        comparison_df["Model"] == model_name
    ]

    plt.plot(
        model_data["K"],
        model_data["NDCG@K"],
        marker="o",
        label=model_name
    )

plt.title("So sánh NDCG@K giữa NCF, TF-IDF Hybrid và BERT Hybrid")
plt.xlabel("K")
plt.ylabel("NDCG@K")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/ncf_tfidf_bert_ndcg_comparison.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 21. HÀM GỢI Ý CHO STREAMLIT
# ------------------------------------------------------------
def recommend_foods_hybrid_bert(
    user_id,
    ncf_model,
    rating_data,
    foods_data,
    num_items,
    user_profiles,
    food_embeddings,
    alpha=0.7,
    top_n=10
):
    """
    Trả về DataFrame món ăn đề xuất bằng Hybrid NCF + BERT.
    """
    seen_items = set(
        rating_data[
            rating_data["user_encoded"] == user_id
        ]["food_encoded"].tolist()
    )

    candidate_items = np.array(
        [
            item_id
            for item_id in range(num_items)
            if item_id not in seen_items
        ],
        dtype=np.int32
    )

    if len(candidate_items) == 0:
        return pd.DataFrame()

    final_scores, ncf_scores, bert_scores = (
        hybrid_bert_scores(
            ncf_model=ncf_model,
            user_id=user_id,
            candidate_items=candidate_items,
            user_profiles=user_profiles,
            food_embeddings=food_embeddings,
            alpha=alpha
        )
    )

    top_indices = np.argsort(
        final_scores
    )[::-1][:top_n]

    top_items = candidate_items[top_indices]

    score_df = pd.DataFrame({
        "food_encoded": top_items,
        "hybrid_bert_score": final_scores[top_indices],
        "ncf_score": ncf_scores[top_indices],
        "bert_content_score": bert_scores[top_indices]
    })

    recommendation_df = score_df.merge(
        foods_data,
        on="food_encoded",
        how="left"
    )

    recommendation_df["rank"] = (
        recommendation_df.index + 1
    )

    return recommendation_df


# ------------------------------------------------------------
# 22. LƯU CẤU HÌNH
# ------------------------------------------------------------
hybrid_bert_config = {
    "best_ncf_model": best_ncf_name,
    "best_alpha_ncf": best_alpha,
    "bert_model_name": bert_config["model_name"],
    "embedding_dimension": int(embedding_dim),
    "num_users": int(num_users),
    "num_items": int(num_items),
    "relevant_threshold": RELEVANT_THRESHOLD,
    "top_liked_items_for_profile": TOP_LIKED_ITEMS,
    "rating_min": RATING_MIN,
    "rating_max": RATING_MAX,
    "food_embeddings_file": (
        "../bert/food_embeddings_normalized.npy"
    ),
    "user_profiles_file": "bert_user_profiles.npy"
}

with open(
    f"{HYBRID_BERT_MODEL_DIR}/hybrid_bert_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        hybrid_bert_config,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n=== HOÀN THÀNH GIAI ĐOẠN 7B ===")
print(
    "Model artifact:",
    HYBRID_BERT_MODEL_DIR
)
print(
    "Kết quả đánh giá:",
    RESULT_DIR
)

