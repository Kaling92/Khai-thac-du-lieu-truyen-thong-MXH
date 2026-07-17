import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ============================================================
# GIAI ĐOẠN 7A - XÂY DỰNG SENTENCE-BERT EMBEDDING CHO MÓN ĂN
#
# Input:
# - processed_data/foods_processed.csv
# - processed_data/food_mapping.csv
#
# Output:
# - models/bert/food_embeddings.npy
# - models/bert/food_embeddings_normalized.npy
# - models/bert/food_embedding_metadata.csv
# - models/bert/bert_config.json
# ============================================================

import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
DATA_DIR = "processed_data"
MODEL_DIR = "models"
BERT_MODEL_DIR = f"{MODEL_DIR}/bert"

os.makedirs(BERT_MODEL_DIR, exist_ok=True)

SEED = 42

# Model đa ngôn ngữ, phù hợp nếu combined_text có tiếng Việt
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Nếu máy có GPU CUDA, code tự dùng GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size:
# - GPU: có thể tăng 32, 64 hoặc 128
# - CPU: nên để 8 hoặc 16
BATCH_SIZE = 32

# ------------------------------------------------------------
# 2. CỐ ĐỊNH RANDOM SEED
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print("=== CẤU HÌNH BERT ===")
print("Model:", MODEL_NAME)
print("Device:", DEVICE)
print("Batch size:", BATCH_SIZE)

# ------------------------------------------------------------
# 3. ĐỌC DỮ LIỆU
# ------------------------------------------------------------
foods_df = pd.read_csv(
    f"{DATA_DIR}/foods_processed.csv"
)

food_mapping_df = pd.read_csv(
    f"{DATA_DIR}/food_mapping.csv"
)

print("\n=== KÍCH THƯỚC DỮ LIỆU ===")
print("foods_processed.csv:", foods_df.shape)
print("food_mapping.csv:", food_mapping_df.shape)

# ------------------------------------------------------------
# 4. KIỂM TRA CỘT BẮT BUỘC
# ------------------------------------------------------------
required_food_columns = [
    "food_id",
    "combined_text"
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
# 5. GHÉP food_encoded VÀO DATASET MÓN ĂN
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

print("\n=== THÔNG TIN MÓN ĂN ===")
print("Số món có embedding:", num_items)

# Kiểm tra food_encoded có liên tục 0 -> num_items - 1
expected_item_ids = set(range(num_items))
actual_item_ids = set(
    foods_df["food_encoded"].unique()
)

if expected_item_ids != actual_item_ids:
    raise ValueError(
        "food_encoded phải liên tục từ 0 đến num_items - 1. "
        "Hãy kiểm tra lại food_mapping.csv."
    )

# ------------------------------------------------------------
# 6. LÀM SẠCH TEXT ĐẦU VÀO CHO BERT
# ------------------------------------------------------------
def clean_text(text):
    """
    Chuẩn hóa text để tránh NaN hoặc chuỗi rỗng.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # BERT vẫn chạy được với text ngắn,
    # nhưng thay text rỗng bằng câu mặc định.
    if len(text) == 0:
        return "Món ăn chưa có mô tả."

    return text


foods_df["bert_input_text"] = (
    foods_df["combined_text"]
    .apply(clean_text)
)

texts = foods_df["bert_input_text"].tolist()

print("\n=== KIỂM TRA TEXT ===")
print("Tổng số text:", len(texts))
print("Ví dụ text đầu tiên:")
print(texts[0][:500])

# ------------------------------------------------------------
# 7. LOAD SENTENCE-BERT MODEL
# ------------------------------------------------------------
print("\nĐang tải Sentence-BERT model...")

bert_model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

print("Đã tải model thành công.")

# ------------------------------------------------------------
# 8. TẠO EMBEDDING CHO TẤT CẢ MÓN ĂN
# ------------------------------------------------------------
print("\nĐang tạo embedding cho món ăn...")

food_embeddings = bert_model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False
)

# Đảm bảo dtype float32 để nhẹ hơn khi lưu
food_embeddings = food_embeddings.astype(
    np.float32
)

print("\n=== THÔNG TIN EMBEDDING ===")
print("Embedding shape:", food_embeddings.shape)
print("Embedding dimension:", food_embeddings.shape[1])

# ------------------------------------------------------------
# 9. NORMALIZE EMBEDDING
# ------------------------------------------------------------
# Khi đã L2-normalize:
# cosine similarity = dot product
embedding_norms = np.linalg.norm(
    food_embeddings,
    axis=1,
    keepdims=True
)

embedding_norms[
    embedding_norms == 0
] = 1e-10

food_embeddings_normalized = (
    food_embeddings / embedding_norms
).astype(np.float32)

print(
    "Đã L2-normalize embedding để dùng cosine similarity."
)

# ------------------------------------------------------------
# 10. KIỂM TRA COSINE SIMILARITY MẪU
# ------------------------------------------------------------
sample_item_id = 0

sample_embedding = food_embeddings_normalized[
    sample_item_id
].reshape(1, -1)

similarity_scores = cosine_similarity(
    sample_embedding,
    food_embeddings_normalized
).flatten()

similarity_scores[sample_item_id] = -1

top_n = 5

top_similar_indices = np.argsort(
    similarity_scores
)[::-1][:top_n]

print("\n=== KIỂM TRA MÓN TƯƠNG TỰ ===")

for rank, item_id in enumerate(
    top_similar_indices,
    start=1
):
    food_row = foods_df[
        foods_df["food_encoded"] == item_id
    ].iloc[0]

    possible_name_columns = [
        "dish_name",
        "food_name",
        "name",
        "title"
    ]

    food_name = None

    for column in possible_name_columns:
        if column in food_row.index:
            food_name = food_row[column]

            if pd.notna(food_name):
                break

    if pd.isna(food_name) or food_name is None:
        food_name = f"Food ID {food_row['food_id']}"

    print(
        f"{rank}. {food_name} "
        f"| Similarity: {similarity_scores[item_id]:.4f}"
    )

# ------------------------------------------------------------
# 11. LƯU EMBEDDING
# ------------------------------------------------------------
np.save(
    f"{BERT_MODEL_DIR}/food_embeddings.npy",
    food_embeddings
)

np.save(
    f"{BERT_MODEL_DIR}/food_embeddings_normalized.npy",
    food_embeddings_normalized
)

# ------------------------------------------------------------
# 12. LƯU METADATA ĐỂ GHÉP VỚI STREAMLIT
# ------------------------------------------------------------
metadata_columns = [
    column
    for column in [
        "food_id",
        "food_encoded",
        "dish_name",
        "food_name",
        "name",
        "title",
        "combined_text",
        "bert_input_text"
    ]
    if column in foods_df.columns
]

food_embedding_metadata = foods_df[
    metadata_columns
].copy()

food_embedding_metadata.to_csv(
    f"{BERT_MODEL_DIR}/food_embedding_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 13. LƯU CẤU HÌNH
# ------------------------------------------------------------
bert_config = {
    "model_name": MODEL_NAME,
    "device_used": DEVICE,
    "batch_size": BATCH_SIZE,
    "num_items": int(num_items),
    "embedding_dimension": int(
        food_embeddings.shape[1]
    ),
    "embedding_file": "food_embeddings.npy",
    "normalized_embedding_file": (
        "food_embeddings_normalized.npy"
    ),
    "metadata_file": "food_embedding_metadata.csv",
    "text_column": "combined_text",
    "normalization": "L2 normalization"
}

with open(
    f"{BERT_MODEL_DIR}/bert_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        bert_config,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n=== HOÀN THÀNH GIAI ĐOẠN 7A ===")
print(f"Embedding được lưu tại: {BERT_MODEL_DIR}")
print("Các file đã tạo:")
print("- food_embeddings.npy")
print("- food_embeddings_normalized.npy")
print("- food_embedding_metadata.csv")
print("- bert_config.json")

