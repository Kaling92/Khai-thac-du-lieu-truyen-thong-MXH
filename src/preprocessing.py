import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.model_selection import train_test_split

def normalize_text(text):
    """
    Cleans and normalizes Vietnamese text.
    """
    if not isinstance(text, str):
        return ""
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Remove emojis and extra whitespace
    text = re.sub(r'[^\s\w,._+-/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_foods(df_foods):
    """
    Cleans food metadata text features and creates content modeling texts.
    """
    df = df_foods.copy()
    
    # Fill missing values
    text_cols = ['dish_name', 'description', 'ingredients', 'cooking_method', 'dish_tags']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
            df[col] = df[col].apply(normalize_text)
            
    # Convert numerical features
    num_cols = ['calories', 'fat', 'fiber', 'sugar', 'protein', 'cooking_time']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Combine text features for Content-Based Filtering (CBF)
    # The teacher's paper uses: ingredients (weight 0.6), description (weight 0.25), dish_tags (weight 0.15)
    # We can model this by repeating text or combining them. Here we concatenate them.
    df['combined_features'] = df['ingredients'] + " " + df['description'] + " " + df['dish_tags']
    return df

def load_and_preprocess_data(foods_path, ratings_path):
    """
    Loads and cleans both datasets.
    """
    df_foods = pd.read_csv(foods_path)
    df_ratings = pd.read_csv(ratings_path)
    
    # Preprocess foods
    df_foods_clean = preprocess_foods(df_foods)
    
    # Preprocess ratings: Keep only the latest rating if duplicates exist (though ratings.csv doesn't have timestamps,
    # we just drop duplicate (userId, foodId) pairs keeping the last one)
    df_ratings_clean = df_ratings.drop_duplicates(subset=['userId', 'foodId'], keep='last')
    
    # Ensure all ratings are floats
    df_ratings_clean['rating'] = df_ratings_clean['rating'].astype(float)
    df_ratings_clean['userId'] = df_ratings_clean['userId'].astype(int)
    df_ratings_clean['foodId'] = df_ratings_clean['foodId'].astype(int)
    
    return df_foods_clean, df_ratings_clean

def split_ratings_data(df_ratings, test_size=0.1, random_state=42):
    """
    Splits the ratings dataset into training and testing sets.
    """
    train_df, test_df = train_test_split(df_ratings, test_size=test_size, random_state=random_state)
    return train_df, test_df

if __name__ == "__main__":
    # Quick sanity check
    f_path = r"d:\Khai thác dữ liệu truyền thông MXH\Clean Dataset-20260614T040023Z-3-001\Clean Dataset\foods.csv"
    r_path = r"d:\Khai thác dữ liệu truyền thông MXH\Clean Dataset-20260614T040023Z-3-001\Clean Dataset\ratings.csv"
    try:
        df_f, df_r = load_and_preprocess_data(f_path, r_path)
        print("Foods shape:", df_f.shape)
        print("Ratings shape:", df_r.shape)
        train_df, test_df = split_ratings_data(df_r)
        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)
    except Exception as e:
        print("Error during sanity check:", str(e))
