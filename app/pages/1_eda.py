import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_raw_data

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .header-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <h1 style="color: white; margin: 0;">Tổng Quan Bộ Dữ Liệu ViFoodRec</h1>
    <p>Phân tích thống kê và khám phá bộ dữ liệu ẩm thực Việt Nam.</p>
</div>
""", unsafe_allow_html=True)

try:
    df_foods, df_ratings = load_raw_data()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng Số Món Ăn", f"{len(df_foods):,}")
    m2.metric("Tổng Số Lượt Đánh Giá", f"{len(df_ratings):,}")
    m3.metric("Số Người Dùng", f"{df_ratings['userId'].nunique():,}")
    density = (len(df_ratings) / (df_ratings['userId'].nunique() * len(df_foods))) * 100
    m4.metric("Độ Đầy Ma Trận", f"{density:.2f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phân Phối Điểm Đánh Giá")
        fig, ax = plt.subplots(figsize=(8, 5))
        rating_counts = df_ratings['rating'].value_counts().sort_index()
        sns.barplot(x=rating_counts.index, y=rating_counts.values, palette='viridis', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Phân Loại Món Ăn")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        type_counts = df_foods['dish_type'].value_counts().head(10)
        sns.barplot(x=type_counts.values, y=type_counts.index, palette='mako', ax=ax2)
        st.pyplot(fig2)
        
    st.markdown("---")
    st.subheader("Danh sách Món ăn (foods.csv)")
    st.dataframe(df_foods.head(50))
except Exception as e:
    st.error(f"Lỗi tải dữ liệu: {e}")
