import pandas as pd
import streamlit as st
from src.config import FOODS_CSV_PATH, RATINGS_CSV_PATH, FOODS_PROCESSED_PATH

@st.cache_data
def load_raw_data():
    df_foods = pd.read_csv(FOODS_CSV_PATH)
    df_ratings = pd.read_csv(RATINGS_CSV_PATH)
    return df_foods, df_ratings

@st.cache_data
def load_processed_data():
    try:
        foods_df = pd.read_csv(FOODS_PROCESSED_PATH)
        return foods_df
    except Exception:
        return pd.DataFrame()
