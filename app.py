import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import our custom recommendation modules
from src.preprocessing import load_and_preprocess_data, split_ratings_data
from src.models import (
    ContentBasedRecommender,
    MemoryCollaborativeFiltering,
    FunkSVDRecommender,
    NCFRecommender
)

# Set page configuration
st.set_page_config(
    page_title="ViFoodRec - Hệ Thống Khuyến Nghị Món Ăn Việt",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design and aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Styling */
    .reportview-container {
        background-color: #F8F9FA;
    }
    
    /* Header Card Styling */
    .header-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2);
    }
    
    .header-card h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .header-card p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Dish Card Styling */
    .dish-card {
        background-color: white;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .dish-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    .dish-title {
        color: #1E3A8A;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .dish-meta {
        font-size: 0.9rem;
        color: #6B7280;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .dish-meta span {
        background-color: #F3F4F6;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
    }
    
    .metric-badge {
        font-size: 1.1rem;
        font-weight: 600;
        color: #10B981;
        float: right;
        background-color: #ECFDF5;
        padding: 0.25rem 1rem;
        border-radius: 8px;
        border: 1px solid #A7F3D0;
    }
    
    .tag-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #2563EB;
        font-size: 0.8rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# Data Loading and Model Caching
# ----------------------------------------
@st.cache_resource
def load_data_and_train_models():
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
    
    df_foods, df_ratings = load_and_preprocess_data(foods_path, ratings_path)
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
    svd = FunkSVDRecommender(n_factors=15, lr=0.005, reg=0.02, epochs=25)
    svd.fit(train_df)
    
    # Train NCF
    ncf = NCFRecommender(embedding_dim=16, layers=[32, 16], epochs=8, batch_size=256, lr=0.001)
    ncf.fit(train_df)
    
    return {
        'df_foods': df_foods,
        'df_ratings': df_ratings,
        'train_df': train_df,
        'test_df': test_df,
        'cbf': cbf,
        'user_cf': user_cf,
        'item_cf': item_cf,
        'svd': svd,
        'ncf': ncf
    }

# Load data and models
with st.spinner("Đang tải dữ liệu và huấn luyện 4 mô hình khuyến nghị... (Vui lòng đợi vài giây)"):
    resources = load_data_and_train_models()

df_foods = resources['df_foods']
df_ratings = resources['df_ratings']
train_df = resources['train_df']
test_df = resources['test_df']

# Sidebar Navigation
st.sidebar.markdown("<h2 style='color:#1E3A8A;'>🍲 ViFoodRec</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Chọn chức năng:", [
    "📊 Tổng Quan Dữ Liệu (EDA)",
    "🔍 Khuyến Nghị Theo Món Ăn (CBF)",
    "👤 Khuyến Nghị Cho Người Dùng"
])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Đồ án môn học:**
Khai thác dữ liệu truyền thông xã hội

**Thực hiện:**
Sinh viên Công nghệ Thông tin
""")

# ----------------------------------------
# 1. Page: EDA
# ----------------------------------------
if page == "📊 Tổng Quan Dữ Liệu (EDA)":
    st.markdown("""
    <div class="header-card">
        <h1>Tổng Quan Bộ Dữ Liệu ViFoodRec</h1>
        <p>Phân tích thống kê và khám phá bộ dữ liệu ẩm thực Việt Nam dùng trong hệ thống khuyến nghị.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng Số Món Ăn (foods.csv)", f"{len(df_foods):,}")
    m2.metric("Tổng Số Lượt Đánh Giá (ratings.csv)", f"{len(df_ratings):,}")
    m3.metric("Số Người Dùng Duy Nhất", f"{df_ratings['userId'].nunique():,}")
    
    # Calculate density
    possible = df_ratings['userId'].nunique() * len(df_foods)
    density = (len(df_ratings) / possible) * 100
    m4.metric("Độ Đầy Ma Trận Ratings (Density)", f"{density:.2f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân Phối Điểm Đánh Giá (Ratings)")
        fig, ax = plt.subplots(figsize=(8, 5))
        rating_counts = df_ratings['rating'].value_counts().sort_index()
        sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis', ax=ax)
        ax.set_xlabel("Điểm Đánh Giá")
        ax.set_ylabel("Số Lượng")
        st.pyplot(fig)
        st.info("Nhận xét: Bộ dữ liệu phân phối điểm đánh giá rất đồng đều giữa các thang điểm từ 0.0 đến 5.0 (gia số 0.5), cho thấy dữ liệu đã được xử lý cân bằng hoặc giả lập đồng bộ để đánh giá mô hình.")
        
    with col2:
        st.subheader("Phân Loại Món Ăn (Dish Types)")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        type_counts = df_foods['dish_type'].value_counts().head(10)
        sns.barplot(x=type_counts.values, y=type_counts.index, palette='mako', ax=ax2)
        ax2.set_xlabel("Số Lượng Món Ăn")
        st.pyplot(fig2)
        
    st.markdown("---")
    st.subheader("Khám phá Danh sách Món ăn")
    st.dataframe(df_foods[['food_id', 'dish_name', 'dish_type', 'serving_size', 'cooking_time', 'calories', 'protein', 'fat']].head(50))


# ----------------------------------------
# 2. Page: CBF (Recommend for Item)
# ----------------------------------------
elif page == "🔍 Khuyến Nghị Theo Món Ăn (CBF)":
    st.markdown("""
    <div class="header-card">
        <h1>Khuyến Nghị Món Ăn Tương Đồng (Content-Based)</h1>
        <p>Tìm kiếm các món ăn có đặc tính, nguyên liệu và mô tả tương tự với món ăn bạn yêu thích.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Select target dish
    dish_list = df_foods.sort_values(by='dish_name')
    dish_options = {row['dish_name']: row['food_id'] for _, row in dish_list.iterrows()}
    
    selected_dish_name = st.selectbox("Chọn món ăn yêu thích của bạn:", list(dish_options.keys()))
    target_food_id = dish_options[selected_dish_name]
    
    # Target dish details
    target_dish = df_foods[df_foods['food_id'] == target_food_id].iloc[0]
    
    # Display selected food details
    t1, t2 = st.columns([1, 2])
    with t1:
        if target_dish['image_link']:
            st.image(target_dish['image_link'], use_container_width=True, caption=target_dish['dish_name'])
        else:
            st.warning("Không có hình ảnh hiển thị.")
            
    with t2:
        st.markdown(f"### {target_dish['dish_name']}")
        st.markdown(f"**Mô tả:** {target_dish['description']}")
        st.markdown(f"**Phân loại:** {target_dish['dish_type']} | **Phục vụ:** {target_dish['serving_size']} | **Thời gian nấu:** {target_dish['cooking_time']} phút")
        st.markdown(f"**Dinh dưỡng:** Lượng Calo: {target_dish['calories']} kcal | Đạm: {target_dish['protein']}g | Béo: {target_dish['fat']}g")
        st.markdown(f"**Nguyên liệu:** {target_dish['ingredients']}")
        
    st.markdown("---")
    
    # Recommender setting
    st.subheader("Bộ Lọc Khuyến Nghị")
    c1, c2, c3 = st.columns(3)
    max_time = c1.slider("Thời gian chế biến tối đa (phút):", 5, 120, 120)
    max_calories = c2.slider("Lượng Calo tối đa (kcal):", 50, 1000, 1000)
    top_k = c3.slider("Số lượng món khuyến nghị:", 5, 20, 10)
    
    # Calculate recommendations
    recs = resources['cbf'].recommend_for_item(target_food_id, top_k=top_k * 2) # Get more to allow filtering
    
    # Apply filters
    filtered_recs = recs[
        (recs['cooking_time'] <= max_time) & 
        (recs['calories'] <= max_calories)
    ].head(top_k)
    
    st.subheader(f"Món ăn tương tự gợi ý cho bạn:")
    if filtered_recs.empty:
        st.info("Không tìm thấy món ăn nào phù hợp với bộ lọc hiện tại.")
    else:
        for idx, row in filtered_recs.iterrows():
            st.markdown(f"""
            <div class="dish-card">
                <span class="metric-badge">Độ tương đồng: {row['similarity_score']:.2%}</span>
                <div class="dish-title">{row['dish_name']}</div>
                <div class="dish-meta">
                    <span><b>Phân loại:</b> {row['dish_type']}</span>
                    <span>⏱️ {row['cooking_time']} phút</span>
                    <span>🔥 {row['calories']} kcal</span>
                    <span>👥 {row['serving_size']}</span>
                </div>
                <div><b>Nguyên liệu:</b> {row['ingredients']}</div>
                <div style="margin-top: 0.5rem;"><b>Tags:</b> 
                    {" ".join([f'<span class="tag-badge">{t.strip()}</span>' for t in row['dish_tags'].split(',') if t])}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ----------------------------------------
# 3. Page: Personalized Recommendations
# ----------------------------------------
elif page == "👤 Khuyến Nghị Cho Người Dùng":
    st.markdown("""
    <div class="header-card">
        <h1>Khuyến Nghị Cá Nhân Hóa (Collaborative Filtering & Deep Learning)</h1>
        <p>Nhập mã người dùng và lựa chọn mô hình để tạo danh sách món ăn gợi ý tốt nhất dựa trên lịch sử đánh giá.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User selection
    user_ids = sorted(df_ratings['userId'].unique())
    selected_user = st.selectbox("Chọn Người Dùng (User ID):", user_ids)
    
    # Model selection
    model_choice = st.selectbox("Chọn Mô Hình Khuyến Nghị:", [
        "User-Based Collaborative Filtering (User-CF)",
        "Item-Based Collaborative Filtering (Item-CF)",
        "Funk SVD (Matrix Factorization)",
        "Neural Collaborative Filtering (NCF - Deep Learning)"
    ])
    
    # Map selection to model resource
    if model_choice == "User-Based Collaborative Filtering (User-CF)":
        model = resources['user_cf']
    elif model_choice == "Item-Based Collaborative Filtering (Item-CF)":
        model = resources['item_cf']
    elif model_choice == "Funk SVD (Matrix Factorization)":
        model = resources['svd']
    else:
        model = resources['ncf']
        
    st.markdown("---")
    
    # Display user's history
    st.subheader("Lịch sử Đánh giá của Người Dùng (Món ăn yêu thích nhất)")
    user_ratings = df_ratings[df_ratings['userId'] == selected_user]
    user_liked = user_ratings.merge(df_foods, left_on='foodId', right_on='food_id')
    user_liked_sorted = user_liked.sort_values(by='rating', ascending=False)
    
    st.dataframe(user_liked_sorted[['dish_name', 'rating', 'dish_type', 'cooking_time', 'calories']].head(10))
    
    st.markdown("---")
    st.subheader("Gợi ý Món Ăn Mới Dành Cho Bạn")
    
    # Find dishes the user hasn't rated yet
    rated_fids = set(user_ratings['foodId'].tolist())
    all_fids = set(df_foods['food_id'].tolist())
    unrated_fids = list(all_fids - rated_fids)
    
    # Predict ratings for unrated foods
    predictions = []
    with st.spinner("Đang tính toán dự báo đánh giá món ăn..."):
        for fid in unrated_fids:
            pred = model.predict_rating(selected_user, fid)
            predictions.append((fid, pred))
            
    # Sort predictions
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
    
    # Show top-10 recommended foods
    for fid, score in predictions:
        dish_info = df_foods[df_foods['food_id'] == fid].iloc[0]
        st.markdown(f"""
        <div class="dish-card">
            <span class="metric-badge" style="color: #3B82F6; background-color: #EFF6FF; border-color: #BFDBFE;">
                Dự báo Rating: {score:.2f} ★
            </span>
            <div class="dish-title">{dish_info['dish_name']}</div>
            <div class="dish-meta">
                <span><b>Phân loại:</b> {dish_info['dish_type']}</span>
                <span>⏱️ {dish_info['cooking_time']} phút</span>
                <span>🔥 {dish_info['calories']} kcal</span>
                <span>👥 {dish_info['serving_size']}</span>
            </div>
            <div><b>Mô tả:</b> {dish_info['description']}</div>
            <div style="margin-top: 0.5rem;"><b>Nguyên liệu:</b> {dish_info['ingredients']}</div>
        </div>
        """, unsafe_allow_html=True)
