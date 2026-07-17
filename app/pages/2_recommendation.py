import streamlit as st
import pandas as pd
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_raw_data
from src.preprocessing import load_and_preprocess_data, split_ratings_data
from src.models import (
    ContentBasedRecommender,
    MemoryCollaborativeFiltering,
    FunkSVDRecommender,
    NCFRecommender
)
from src.config import FOODS_CSV_PATH, RATINGS_CSV_PATH

st.set_page_config(page_title="Recommendation", page_icon="👤", layout="wide")

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
    .dish-meta span { background-color: #F3F4F6; padding: 4px 10px; border-radius: 20px; margin-right: 10px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_and_train_models():
    df_foods, df_ratings = load_and_preprocess_data(FOODS_CSV_PATH, RATINGS_CSV_PATH)
    train_df, test_df = split_ratings_data(df_ratings, test_size=0.1, random_state=42)
    
    # Train Content-Based
    cbf = ContentBasedRecommender()
    cbf.fit(df_foods, train_df)
    
    # Train User-CF & Item-CF
    user_cf = MemoryCollaborativeFiltering(kind='user', similarity='cosine', k=15)
    user_cf.fit(train_df)
    
    item_cf = MemoryCollaborativeFiltering(kind='item', similarity='cosine', k=15)
    item_cf.fit(train_df)
    
    # Train Funk SVD
    svd = FunkSVDRecommender(n_factors=10, lr=0.01, reg=0.02, epochs=8)
    svd.fit(train_df)
    
    # Train NCF (Warning: this might be slow to re-train on the fly, normally we load saved model)
    ncf = NCFRecommender(embedding_dim=16, layers=[32, 16], epochs=3, batch_size=256, lr=0.001)
    ncf.fit(train_df)
    
    return {
        'df_foods': df_foods,
        'df_ratings': df_ratings,
        'train_df': train_df,
        'cbf': cbf,
        'user_cf': user_cf,
        'item_cf': item_cf,
        'svd': svd,
        'ncf': ncf
    }

st.markdown("""
<div class="header-card">
    <h1 style="color: white; margin: 0;">Khuyến Nghị Món Ăn</h1>
    <p>Chọn mô hình và người dùng để nhận gợi ý món ăn (CBF, CF, SVD, NCF).</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Đang khởi tạo các mô hình Recommendation..."):
    try:
        resources = load_and_train_models()
        df_foods = resources['df_foods']
        df_ratings = resources['df_ratings']
        
        tab1, tab2 = st.tabs(["🔍 Khuyến Nghị Theo Món (CBF)", "👤 Khuyến Nghị Cá Nhân Hóa (CF, SVD, NCF)"])
        
        with tab1:
            st.subheader("Tìm món tương tự (Content-Based)")
            dish_list = df_foods.sort_values(by='dish_name')
            dish_options = {row['dish_name']: row['food_id'] for _, row in dish_list.iterrows()}
            selected_dish_name = st.selectbox("Chọn món ăn yêu thích:", list(dish_options.keys()))
            target_food_id = dish_options[selected_dish_name]
            
            top_k = st.slider("Số lượng gợi ý:", 5, 20, 10, key="cbf_slider")
            
            recs = resources['cbf'].recommend_for_item(target_food_id, top_k=top_k)
            for idx, row in recs.iterrows():
                st.markdown(f"""
                <div class="dish-card">
                    <span class="metric-badge">Độ tương đồng: {row['similarity_score']:.2%}</span>
                    <div class="dish-title">{row['dish_name']}</div>
                    <div class="dish-meta">
                        <span><b>Phân loại:</b> {row['dish_type']}</span>
                        <span>⏱️ {row['cooking_time']} phút</span>
                        <span>🔥 {row['calories']} kcal</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with tab2:
            user_ids = sorted(df_ratings['userId'].unique())
            selected_user = st.selectbox("Chọn Người Dùng (User ID):", user_ids)
            model_choice = st.selectbox("Chọn Mô Hình:", [
                "User-CF", "Item-CF", "Funk SVD", "NCF"
            ])
            
            model = resources['user_cf']
            if model_choice == "Item-CF": model = resources['item_cf']
            elif model_choice == "Funk SVD": model = resources['svd']
            elif model_choice == "NCF": model = resources['ncf']
            
            if st.button("Lấy Gợi Ý", type="primary"):
                with st.spinner("Đang phân tích..."):
                    user_ratings = df_ratings[df_ratings['userId'] == selected_user]
                    rated_fids = set(user_ratings['foodId'].tolist())
                    all_fids = set(df_foods['food_id'].tolist())
                    unrated_fids = list(all_fids - rated_fids)
                    
                    predictions = []
                    for fid in unrated_fids:
                        pred = model.predict_rating(selected_user, fid)
                        predictions.append((fid, pred))
                        
                    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
                    
                    for fid, score in predictions:
                        dish_info = df_foods[df_foods['food_id'] == fid].iloc[0]
                        st.markdown(f"""
                        <div class="dish-card">
                            <span class="metric-badge" style="color: #3B82F6; background-color: #EFF6FF;">
                                Rating Dự báo: {score:.2f} ★
                            </span>
                            <div class="dish-title">{dish_info['dish_name']}</div>
                            <div class="dish-meta">
                                <span><b>Phân loại:</b> {dish_info['dish_type']}</span>
                                <span>⏱️ {dish_info['cooking_time']} phút</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}")
