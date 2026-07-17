import streamlit as st
import os
import sys

# Ensure root is in path so we can import src
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="ViFoodRec - Hệ Thống Khuyến Nghị Món Ăn",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .header-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white; padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem;
    }
    .header-card h1 { font-size: 2.5rem; font-weight: 700; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🍲 ViFoodRec")
st.sidebar.markdown("---")

st.markdown("""
<div class="header-card">
    <h1>ViFoodRec Dashboard</h1>
    <p>Hệ thống Khuyến nghị Món ăn Việt Nam. Vui lòng chọn một tính năng từ Menu bên trái.</p>
</div>
""", unsafe_allow_html=True)

st.info("👈 Hãy chọn một trang trên Sidebar để bắt đầu (EDA, Recommendation, Semantic Search...).")
