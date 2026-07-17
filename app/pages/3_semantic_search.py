import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_processed_data
from src.config import BERT_MODEL_DIR

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .header-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
    }
    .dish-card {
        background-color: white; border-radius: 12px; border: 1px solid #E5E7EB;
        padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .metric-badge { color: #10B981; float: right; font-weight: bold; background-color: #ECFDF5; padding: 4px 10px; border-radius: 8px;}
    .dish-title { color: #1E3A8A; font-size: 1.4rem; font-weight: 600; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <h1 style="color: white; margin: 0;">🔍 Semantic Search & Tương Tự BERT</h1>
    <p>Tìm kiếm món ăn dựa trên Vector ngữ nghĩa bằng mô hình Sentence-BERT.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bert_data():
    foods_df = load_processed_data()
    emb_path = os.path.join(BERT_MODEL_DIR, "food_embeddings_normalized.npy")
    if os.path.exists(emb_path):
        food_embeddings = np.load(emb_path).astype(np.float32)
    else:
        food_embeddings = None
    return foods_df, food_embeddings

foods_df, food_embeddings = load_bert_data()

if food_embeddings is None:
    st.warning("⚠️ Không tìm thấy file `food_embeddings_normalized.npy` trong thư mục `models/bert`. Bạn cần chạy pipeline SBERT trước.")
else:
    if not foods_df.empty and 'food_encoded' in foods_df.columns:
        dish_list = foods_df.sort_values(by='dish_name' if 'dish_name' in foods_df.columns else foods_df.columns[1])
        dish_options = {row.get('dish_name', f"Dish {row['food_id']}"): row['food_encoded'] for _, row in dish_list.iterrows()}
        
        selected_dish_name = st.selectbox("Chọn một món ăn:", list(dish_options.keys()))
        selected_encoded = int(dish_options[selected_dish_name])
        top_n = st.slider("Số lượng món tương tự:", 5, 20, 10)
        
        if st.button("Tìm Kiếm", type="primary"):
            with st.spinner("Đang trích xuất đặc trưng ngữ nghĩa..."):
                selected_embedding = food_embeddings[selected_encoded]
                similarity_scores = np.dot(food_embeddings, selected_embedding)
                similarity_scores[selected_encoded] = -1
                
                top_indices = np.argsort(similarity_scores)[::-1][:top_n]
                
                result_df = foods_df[foods_df["food_encoded"].isin(top_indices)].copy()
                similarity_mapping = {int(fid): float(similarity_scores[fid]) for fid in top_indices}
                result_df["bert_similarity"] = result_df["food_encoded"].map(similarity_mapping)
                result_df = result_df.sort_values("bert_similarity", ascending=False).reset_index(drop=True)
                
                for idx, row in result_df.iterrows():
                    dish_name = row.get('dish_name', f"Món ăn ID: {row['food_id']}")
                    desc = row.get('description', '')
                    st.markdown(f"""
                    <div class="dish-card">
                        <span class="metric-badge">BERT Score: {row['bert_similarity']:.4f}</span>
                        <div class="dish-title">{dish_name}</div>
                        <div>{desc[:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.error("Dữ liệu foods_processed.csv không hợp lệ hoặc thiếu cột.")
