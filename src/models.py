import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# ----------------------------------------
# 1. Content-Based Recommender
# ----------------------------------------
class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=2000)
        self.tfidf_matrix = None
        self.df_foods = None
        self.food_id_to_idx = {}
        self.user_profiles = {}
        self.global_mean = 2.5
        
    def fit(self, df_foods, train_df=None):
        self.df_foods = df_foods.copy().reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_foods['combined_features'])
        self.food_id_to_idx = {
            int(fid): idx for idx, fid in enumerate(self.df_foods['food_id'].values)
        }
        self.train_df = train_df
        self.global_mean = train_df['rating'].mean() if train_df is not None else 2.5
        self._build_user_profiles()
        
    def _build_user_profiles(self):
        """Precompute weighted TF-IDF profiles per user for fast inference."""
        self.user_profiles = {}
        if self.train_df is None:
            return
            
        for uid, group in self.train_df.groupby('userId'):
            liked = group[group['rating'] >= 3.5]
            if liked.empty:
                liked = group.sort_values('rating', ascending=False).head(5)
                
            indices = []
            weights = []
            for fid, rating in zip(liked['foodId'].values, liked['rating'].values):
                idx = self.food_id_to_idx.get(int(fid))
                if idx is not None:
                    indices.append(idx)
                    weights.append(float(rating))
                    
            if not indices:
                self.user_profiles[int(uid)] = None
                continue
                
            profile = np.zeros((1, self.tfidf_matrix.shape[1]))
            for idx, weight in zip(indices, weights):
                profile += self.tfidf_matrix[idx].toarray() * weight
            profile /= sum(weights)
            self.user_profiles[int(uid)] = profile
        
    def predict_rating(self, user_id, food_id):
        if self.train_df is None:
            return self.global_mean
            
        idx = self.food_id_to_idx.get(int(food_id))
        if idx is None:
            return self.global_mean
            
        profile = self.user_profiles.get(int(user_id))
        if profile is None:
            user_ratings = self.train_df[self.train_df['userId'] == user_id]
            return float(user_ratings['rating'].mean()) if not user_ratings.empty else self.global_mean
            
        sim = cosine_similarity(profile, self.tfidf_matrix[idx]).flatten()[0]
        pred = max(sim, 0.0) * 5.0
        return float(np.clip(pred, 0.0, 5.0))
        
    def recommend_for_item(self, food_id, top_k=10):
        """
        Recommends dishes similar to a given food item.
        """
        idx_list = self.df_foods.index[self.df_foods['food_id'] == food_id].tolist()
        if not idx_list:
            return pd.DataFrame()
        idx = idx_list[0]
        
        # Calculate cosine similarity between this food and all others
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Sort scores
        related_indices = np.argsort(sim_scores)[::-1]
        
        # Exclude the queried food item itself
        related_indices = [i for i in related_indices if i != idx]
        
        recommendations = self.df_foods.iloc[related_indices[:top_k]].copy()
        recommendations['similarity_score'] = sim_scores[related_indices[:top_k]]
        return recommendations
        
    def recommend_for_user(self, user_ratings, top_k=10):
        """
        Recommends dishes for a user based on their high-rated foods.
        user_ratings: DataFrame containing user's ratings with columns 'foodId' and 'rating'
        """
        # Get highly rated foods (rating >= 3.5)
        liked_ratings = user_ratings[user_ratings['rating'] >= 3.5]
        if liked_ratings.empty:
            # Fallback to all rated foods sorted by rating
            liked_ratings = user_ratings.sort_values(by='rating', ascending=False)
            if liked_ratings.empty:
                return self.df_foods.head(top_k)
                
        liked_food_ids = liked_ratings['foodId'].tolist()
        liked_weights = liked_ratings['rating'].values
        
        # Map food_ids to matrix indices
        indices = []
        valid_weights = []
        for fid, weight in zip(liked_food_ids, liked_weights):
            idx_list = self.df_foods.index[self.df_foods['food_id'] == fid].tolist()
            if idx_list:
                indices.append(idx_list[0])
                valid_weights.append(weight)
                
        if not indices:
            return self.df_foods.head(top_k)
            
        # Calculate user profile as weighted average of food vectors
        user_profile = np.zeros((1, self.tfidf_matrix.shape[1]))
        for idx, w in zip(indices, valid_weights):
            user_profile += self.tfidf_matrix[idx].toarray() * w
        user_profile /= sum(valid_weights)
        
        # Calculate similarity between user profile and all foods
        sim_scores = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        
        # Exclude foods the user has already rated
        rated_fids = set(user_ratings['foodId'].tolist())
        
        sorted_indices = np.argsort(sim_scores)[::-1]
        recommended_indices = []
        for idx in sorted_indices:
            fid = self.df_foods.iloc[idx]['food_id']
            if fid not in rated_fids:
                recommended_indices.append(idx)
                if len(recommended_indices) == top_k:
                    break
                    
        recommendations = self.df_foods.iloc[recommended_indices].copy()
        recommendations['similarity_score'] = sim_scores[recommended_indices]
        return recommendations


# ----------------------------------------
# 2. Memory-Based Collaborative Filtering (User-CF & Item-CF)
# ----------------------------------------
class MemoryCollaborativeFiltering:
    def __init__(self, kind='user', similarity='cosine', k=10):
        self.kind = kind  # 'user' or 'item'
        self.similarity = similarity  # 'cosine' or 'pearson'
        self.k = k  # number of neighbors
        self.rating_matrix = None
        self.user_means = None
        self.sim_matrix = None
        self.user_map = {}
        self.item_map = {}
        self.inv_user_map = {}
        self.inv_item_map = {}
        
    def fit(self, train_df):
        # Create user and item mappings
        unique_users = sorted(train_df['userId'].unique())
        unique_items = sorted(train_df['foodId'].unique())
        
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {iid: j for j, iid in enumerate(unique_items)}
        self.inv_user_map = {i: uid for uid, i in self.user_map.items()}
        self.inv_item_map = {j: iid for iid, j in self.item_map.items()}
        
        # Create sparse rating matrix
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        rows = train_df['userId'].map(self.user_map).values
        cols = train_df['foodId'].map(self.item_map).values
        ratings = train_df['rating'].values
        
        # Dense rating matrix for calculations (since dataset is relatively small: 101 x 4000)
        self.rating_matrix = np.zeros((num_users, num_items))
        self.rating_matrix[rows, cols] = ratings
        
        # Calculate user means (only for rated items, ratings > 0)
        self.user_means = np.zeros(num_users)
        for u in range(num_users):
            rated = self.rating_matrix[u] > 0
            if np.any(rated):
                self.user_means[u] = np.mean(self.rating_matrix[u, rated])
            else:
                self.user_means[u] = 0.0
                
        # Compute similarities
        if self.kind == 'user':
            # Center ratings for Pearson
            if self.similarity == 'pearson':
                centered = np.zeros_like(self.rating_matrix)
                for u in range(num_users):
                    rated = self.rating_matrix[u] > 0
                    centered[u, rated] = self.rating_matrix[u, rated] - self.user_means[u]
                self.sim_matrix = cosine_similarity(centered)
            else:
                self.sim_matrix = cosine_similarity(self.rating_matrix)
        else:
            # Item similarity
            if self.similarity == 'pearson':
                # Center by item means
                centered = np.zeros_like(self.rating_matrix)
                for i in range(num_items):
                    rated = self.rating_matrix[:, i] > 0
                    if np.any(rated):
                        item_mean = np.mean(self.rating_matrix[rated, i])
                        centered[rated, i] = self.rating_matrix[rated, i] - item_mean
                self.sim_matrix = cosine_similarity(centered.T)
            else:
                self.sim_matrix = cosine_similarity(self.rating_matrix.T)
                
        # Fill self-similarity with 0 to prevent recommending items based on themselves
        np.fill_diagonal(self.sim_matrix, 0)
        
    def predict_rating(self, user_id, food_id):
        # Fallback to global mean if user or item is unseen
        global_mean = 2.5
        if user_id not in self.user_map and food_id not in self.item_map:
            return global_mean
        if user_id not in self.user_map:
            # Item average rating
            j = self.item_map[food_id]
            rated_users = self.rating_matrix[:, j] > 0
            return np.mean(self.rating_matrix[rated_users, j]) if np.any(rated_users) else global_mean
        if food_id not in self.item_map:
            # User average rating
            u = self.user_map[user_id]
            return self.user_means[u] if self.user_means[u] > 0 else global_mean
            
        u = self.user_map[user_id]
        j = self.item_map[food_id]
        
        if self.kind == 'user':
            # User-based prediction
            # Find users who rated item j
            rated_users = np.where(self.rating_matrix[:, j] > 0)[0]
            if len(rated_users) == 0:
                return self.user_means[u] if self.user_means[u] > 0 else global_mean
                
            # Get similarity of these users to user u
            sims = self.sim_matrix[u, rated_users]
            
            # Take top-k similar users
            top_k_idx = np.argsort(sims)[::-1][:self.k]
            top_k_users = rated_users[top_k_idx]
            top_k_sims = sims[top_k_idx]
            
            sim_sum = np.sum(np.abs(top_k_sims))
            if sim_sum == 0:
                return self.user_means[u]
                
            ratings = self.rating_matrix[top_k_users, j]
            means = self.user_means[top_k_users]
            
            # Predict
            pred = self.user_means[u] + np.sum(top_k_sims * (ratings - means)) / sim_sum
            return np.clip(pred, 0.0, 5.0)
        else:
            # Item-based prediction
            # Find items rated by user u
            rated_items = np.where(self.rating_matrix[u] > 0)[0]
            if len(rated_items) == 0:
                return global_mean
                
            # Get similarity of these items to item j
            sims = self.sim_matrix[j, rated_items]
            
            # Take top-k similar items
            top_k_idx = np.argsort(sims)[::-1][:self.k]
            top_k_items = rated_items[top_k_idx]
            top_k_sims = sims[top_k_idx]
            
            sim_sum = np.sum(np.abs(top_k_sims))
            if sim_sum == 0:
                # Fallback to item average rating
                rated_users = self.rating_matrix[:, j] > 0
                return np.mean(self.rating_matrix[rated_users, j]) if np.any(rated_users) else global_mean
                
            ratings = self.rating_matrix[u, top_k_items]
            pred = np.sum(top_k_sims * ratings) / sim_sum
            return np.clip(pred, 0.0, 5.0)


# ----------------------------------------
# 3. Model-Based Collaborative Filtering (Funk SVD Matrix Factorization)
# ----------------------------------------
class FunkSVDRecommender:
    def __init__(self, n_factors=20, lr=0.005, reg=0.02, epochs=30):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        
        self.user_map = {}
        self.item_map = {}
        self.global_mean = 0.0
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None
        
    def fit(self, train_df):
        unique_users = sorted(train_df['userId'].unique())
        unique_items = sorted(train_df['foodId'].unique())
        
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {iid: j for j, iid in enumerate(unique_items)}
        
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        # Initialize biases and latent matrices
        self.global_mean = train_df['rating'].mean()
        self.b_u = np.zeros(num_users)
        self.b_i = np.zeros(num_items)
        
        # Initialize latent vectors with small random values (seed for reproducibility)
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (num_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (num_items, self.n_factors))
        
        # Extract train samples
        u_indices = train_df['userId'].map(self.user_map).values
        i_indices = train_df['foodId'].map(self.item_map).values
        ratings = train_df['rating'].values
        
        # SGD optimization loop (shuffle once per epoch)
        n_samples = len(ratings)
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            for idx in indices:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]
                
                pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                
                p_temp = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_temp - self.reg * self.Q[i])
                
    def batch_predict(self, user_ids, food_ids):
        preds = np.full(len(user_ids), self.global_mean, dtype=float)
        for idx, (user_id, food_id) in enumerate(zip(user_ids, food_ids)):
            if user_id in self.user_map and food_id in self.item_map:
                u = self.user_map[user_id]
                i = self.item_map[food_id]
                preds[idx] = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
            elif user_id in self.user_map:
                u = self.user_map[user_id]
                preds[idx] = self.global_mean + self.b_u[u]
            elif food_id in self.item_map:
                i = self.item_map[food_id]
                preds[idx] = self.global_mean + self.b_i[i]
        return np.clip(preds, 0.0, 5.0)
                
    def predict_rating(self, user_id, food_id):
        # Handle cold start / unseen users and items
        if user_id not in self.user_map and food_id not in self.item_map:
            return self.global_mean
        if user_id not in self.user_map:
            if food_id in self.item_map:
                i = self.item_map[food_id]
                return self.global_mean + self.b_i[i]
            return self.global_mean
        if food_id not in self.item_map:
            u = self.user_map[user_id]
            return self.global_mean + self.b_u[u]
            
        u = self.user_map[user_id]
        i = self.item_map[food_id]
        pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return np.clip(pred, 0.0, 5.0)


# ----------------------------------------
# 4. Neural Collaborative Filtering (NCF)
# ----------------------------------------
from sklearn.neural_network import MLPRegressor

class SklearnNCFRecommender:
    """Fallback neural recommender using scikit-learn MLP when PyTorch is unavailable."""
    def __init__(self, hidden_layers=(32, 16), epochs=50, batch_size=256, lr=0.001):
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.user_map = {}
        self.item_map = {}
        self.global_mean = 2.5
        self.model = None

    def fit(self, train_df):
        unique_users = sorted(train_df['userId'].unique())
        unique_items = sorted(train_df['foodId'].unique())
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.item_map = {iid: j for j, iid in enumerate(unique_items)}
        self.global_mean = float(train_df['rating'].mean())

        num_users = len(unique_users)
        num_items = len(unique_items)
        X = np.column_stack([
            train_df['userId'].map(self.user_map).values / max(num_users, 1),
            train_df['foodId'].map(self.item_map).values / max(num_items, 1),
        ])
        y = train_df['rating'].values

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=min(self.batch_size, len(y)),
            learning_rate_init=self.lr,
            max_iter=self.epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.model.fit(X, y)

    def batch_predict(self, user_ids, food_ids):
        num_users = len(self.user_map) or 1
        num_items = len(self.item_map) or 1
        X = np.column_stack([
            np.array([
                self.user_map.get(int(uid), 0) / num_users for uid in user_ids
            ]),
            np.array([
                self.item_map.get(int(fid), 0) / num_items for fid in food_ids
            ]),
        ])
        preds = self.model.predict(X) if self.model is not None else np.full(len(user_ids), self.global_mean)
        return np.clip(preds, 0.0, 5.0)

    def predict_rating(self, user_id, food_id):
        return float(self.batch_predict([user_id], [food_id])[0])


def _build_ncf_recommender(embedding_dim=16, layers=None, epochs=8, batch_size=256, lr=0.001):
    layers = layers or [32, 16]
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
    except OSError:
        print("  [WARN] PyTorch unavailable (memory). Using sklearn MLP fallback for Neural CF.")
        return SklearnNCFRecommender(hidden_layers=tuple(layers), epochs=max(epochs * 5, 30), batch_size=batch_size, lr=lr)

    class NCFDataset(Dataset):
        def __init__(self, user_indices, item_indices, ratings):
            self.user_indices = torch.tensor(user_indices, dtype=torch.long)
            self.item_indices = torch.tensor(item_indices, dtype=torch.long)
            self.ratings = torch.tensor(ratings, dtype=torch.float)

        def __len__(self):
            return len(self.ratings)

        def __getitem__(self, idx):
            return self.user_indices[idx], self.item_indices[idx], self.ratings[idx]

    class NCFNet(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim=16, layers=[32, 16, 8]):
            super(NCFNet, self).__init__()
            self.user_embed = nn.Embedding(num_users, embedding_dim)
            self.item_embed = nn.Embedding(num_items, embedding_dim)
            mlp_modules = []
            input_dim = embedding_dim * 2
            for output_dim in layers:
                mlp_modules.append(nn.Linear(input_dim, output_dim))
                mlp_modules.append(nn.ReLU())
                mlp_modules.append(nn.Dropout(0.2))
                input_dim = output_dim
            self.mlp = nn.Sequential(*mlp_modules)
            self.prediction_layer = nn.Linear(input_dim, 1)

        def forward(self, user_input, item_input):
            user_emb = self.user_embed(user_input)
            item_emb = self.item_embed(item_input)
            x = torch.cat([user_emb, item_emb], dim=-1)
            x = self.mlp(x)
            return self.prediction_layer(x).squeeze(-1)

    class TorchNCFRecommender:
        def __init__(self, embedding_dim=16, layers=None, epochs=10, batch_size=256, lr=0.001):
            self.embedding_dim = embedding_dim
            self.layers = layers or [32, 16]
            self.epochs = epochs
            self.batch_size = batch_size
            self.lr = lr
            self.user_map = {}
            self.item_map = {}
            self.global_mean = 0.0
            self.model = None
            self.device = torch.device("cpu")

        def fit(self, train_df):
            import torch
            import random
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            unique_users = sorted(train_df['userId'].unique())
            unique_items = sorted(train_df['foodId'].unique())
            self.user_map = {uid: i for i, uid in enumerate(unique_users)}
            self.item_map = {iid: j for j, iid in enumerate(unique_items)}
            num_users = len(unique_users)
            num_items = len(unique_items)
            self.global_mean = train_df['rating'].mean()

            u_indices = train_df['userId'].map(self.user_map).values
            i_indices = train_df['foodId'].map(self.item_map).values
            ratings = train_df['rating'].values
            dataset = NCFDataset(u_indices, i_indices, ratings)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            self.model = NCFNet(num_users, num_items, self.embedding_dim, self.layers)
            self.model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            self.model.train()
            for _ in range(self.epochs):
                for users, items, targets in dataloader:
                    users = users.to(self.device)
                    items = items.to(self.device)
                    targets = targets.to(self.device)
                    optimizer.zero_grad()
                    predictions = self.model(users, items)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()

        def batch_predict(self, user_ids, food_ids):
            if self.model is None:
                return np.full(len(user_ids), self.global_mean, dtype=float)
            preds = np.full(len(user_ids), self.global_mean, dtype=float)
            valid_u, valid_i, valid_idx = [], [], []
            for idx, (user_id, food_id) in enumerate(zip(user_ids, food_ids)):
                if user_id in self.user_map and food_id in self.item_map:
                    valid_u.append(self.user_map[user_id])
                    valid_i.append(self.item_map[food_id])
                    valid_idx.append(idx)
            if not valid_idx:
                return preds
            self.model.eval()
            with torch.no_grad():
                u_t = torch.tensor(valid_u, dtype=torch.long).to(self.device)
                i_t = torch.tensor(valid_i, dtype=torch.long).to(self.device)
                batch_preds = self.model(u_t, i_t).cpu().numpy()
                preds[valid_idx] = batch_preds
            return np.clip(preds, 0.0, 5.0)

        def predict_rating(self, user_id, food_id):
            return float(self.batch_predict([user_id], [food_id])[0])

    return TorchNCFRecommender(embedding_dim=embedding_dim, layers=layers, epochs=epochs, batch_size=batch_size, lr=lr)


class NCFRecommender:
    def __init__(self, embedding_dim=16, layers=None, epochs=10, batch_size=256, lr=0.001):
        self._impl = _build_ncf_recommender(embedding_dim, layers, epochs, batch_size, lr)

    def fit(self, train_df):
        return self._impl.fit(train_df)

    def batch_predict(self, user_ids, food_ids):
        return self._impl.batch_predict(user_ids, food_ids)

    def predict_rating(self, user_id, food_id):
        return self._impl.predict_rating(user_id, food_id)
