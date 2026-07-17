# CẨM NANG SƯ PHẠM VÀ HƯỚNG DẪN KỸ THUẬT: HỆ THỐNG GỢI Ý VIFOODREC (2026)
*Tài liệu hướng dẫn chuyên sâu từ cơ bản đến nâng cao về kiến trúc thuật toán, độ đo đánh giá và phân tích hiệu năng thực nghiệm.*

---

## 🌟 LỜI MỞ ĐẦU & TỔNG QUAN HỆ THỐNG
Chào mừng bạn đến với Cẩm nang Sư phạm dành cho dự án **ViFoodRec (2026)**. Tài liệu này được biên soạn bởi Cố vấn Học thuật và Kiến trúc sư Phần mềm Cấp cao, nhằm giúp bạn nắm vững bản chất cốt lõi của mọi thuật toán khuyến nghị trong hệ thống, từ các bộ lọc cổ điển cho đến mạng học sâu phức tạp. 

Tài liệu được chia làm 3 tầng giải thích cho mỗi khái niệm:
1. **ELI5 (Explain Like I'm 5):** Giải thích bằng hình tượng đời thường, không thuật ngữ kỹ thuật, thích hợp để thuyết trình hoặc trả lời vấn đề cốt lõi khi bảo vệ luận văn.
2. **Công thức Toán học (The Math):** Công thức toán học chuẩn tắc của thuật toán.
3. **Phân tích Mã nguồn (Code Trace):** Giải thích cách dòng code Python thực tế tính toán từ ma trận dữ liệu.

---

## 📊 BẢNG SO SÁNH PHÁT TRIỂN: VIFOODREC 2024 VS 2026

Dưới đây là ma trận đối chiếu 10 chiều giữa dự án cơ sở (2024) và đề tài mở rộng (2026) được cài đặt thực tế trong codebase của bạn:

| Chiều so sánh | Dự án Cơ sở (ViFoodRec 2024) | Nghiên cứu Mở rộng (ViFoodRec 2026) | Trạng thái Xác minh trong Code |
| :--- | :--- | :--- | :--- |
| **1. Bản chất cốt lõi** | Khai phá sơ khởi hệ thống gợi ý món ăn dựa trên luật tuyến tính đơn giản. | Tối ưu & lấp đầy khoảng trống tuyến tính bằng cách cho Phân rã ma trận và Học sâu đối đầu trực tiếp. | **Đã hoàn thành** (`models.py` cài đặt FunkSVD và NCF) |
| **2. Bản chất dữ liệu** | Dữ liệu thô cào từ Web, nhiều nhiễu, phân tán. | Clean Dataset chuẩn hóa, mật độ cực cao: 101 users x 4000 món, 182.678 tương tác (Density: 45.22%). | **Đã hoàn thành** (`preprocessing.py` tải đúng file sạch) |
| **3. Tiền xử lý** | CBF xóa trùng tên món ăn; CF giữ rating cuối nếu trùng; điền khuyết 40% bằng Median. | Ghép trường văn bản cho CBF: `ingredients + description + dish_tags`. Vector hóa TF-IDF (max_features=2000). | **Đã hoàn thành** (`preprocessing.py` ghép chuỗi, `models.py` TF-IDF) |
| **4. Kiến trúc thuật toán** | Mô hình tuyến tính đơn giản: CBF (5 độ đo đơn lẻ & 1 composite); Memory-CF (User-CF, Item-CF với Cosine/Pearson). | Kết hợp lai cấu trúc: CBF, Memory-CF, Funk SVD (Matrix Factorization với SGD), Neural CF (Deep Learning với MLP). | **Đã hoàn thành** (`models.py` cài đặt đủ 4 nhóm mô hình) |
| **5. Công cụ & Thư viện** | Selenium, BeautifulSoup, Streamlit chạy cục bộ. | Python 3.12+, scikit-learn, PyTorch (có fallback sang Sklearn MLPRegressor nếu thiếu RAM/GPU). | **Đã hoàn thành** (`requirements.txt` & fallback trong `models.py`) |
| **6. Thiết kế thí nghiệm** | Thử nghiệm thủ công trên 200 món ăn (5% dữ liệu) cho CBF; 200 mẫu test cho CF. | Phân chia ngẫu nhiên 90% Train / 10% Test cố định với `random_state=42`. Ngưỡng liên quan (relevance threshold) $\ge 3.5$. | **Đã hoàn thành** (`train.py` & `preprocessing.py` khóa seed 42) |
| **7. Đầu ra hệ thống** | Dự đoán điểm rating thô và gợi ý danh sách món ăn tương tự. | Song song 2 tác vụ: Regression (dự đoán rating $[0, 5]$) và Ranking (Gợi ý Top-10 món ăn chưa tương tác). | **Đã hoàn thành** (`train.py` đánh giá song song 2 tác vụ) |
| **8. Độ đo đánh giá** | CBF: Precision@K, MRR. CF: MSE, RMSE, MAE, NMAE. | Tính toán đồng thời 8 độ đo: Regression (MSE, RMSE, MAE, NMAE) và Ranking (Precision@10, Recall@10, NDCG@10, MRR). | **Đã hoàn thành** (`evaluate.py` cài đặt đủ 8 công thức) |
| **9. Kết quả thực nghiệm** | CBF composite thắng; CF Cosine RMSE=2.0635, Item-CF Cosine MAE=1.6902. | NCF vô địch Regression (RMSE=1.5942); User-CF Cosine dẫn đầu NDCG@10 (39.92%); Funk SVD đạt MRR tốt nhất (60.92%). | **Đã hoàn thành** (Đã đồng bộ hóa LaTeX, Slides và Code) |
| **10. Phân tích lỗi** | Nhận xét chung chung về nhiễu dữ liệu thu thập và tính chủ quan của người dùng. | Phân tích sâu 5 khía cạnh: Cold-start, Popularity bias, Ma trận dẹt mất ổn định Item-CF, Rating ambiguity ngưỡng 3.5. | **Đã hoàn thành** (Chi tiết hóa trong `report.tex` mục 6) |

---

## 🍲 PHẦN 1: CONTENT-BASED FILTERING (CBF) VỚI TF-IDF

### 1. ELI5 (Giải thích trực quan)
Hãy tưởng tượng bạn là một thủ thư. Một người đến mượn sách nói: *"Tôi rất thích cuốn sách có nguyên liệu là thịt bò, hành tây và có mô tả là món xào cay nồng"*. 
Để tìm cuốn sách tiếp theo cho họ, bạn sẽ đi xem tất cả các cuốn sách nấu ăn khác, đếm xem cuốn nào có nhiều từ "thịt bò", "hành tây", "cay nồng" nhất để giới thiệu. Tuy nhiên, nếu bạn chỉ đếm từ thông thường, những từ xuất hiện quá nhiều ở mọi cuốn sách như "thực phẩm", "nấu ăn", "bước" sẽ làm loãng kết quả. TF-IDF chính là chiếc kính lọc giúp bạn **bỏ qua những từ phổ biến vô nghĩa** và **tập trung vào các từ đặc trưng độc bản** của món ăn đó.

### 2. Công thức Toán học (The Math)
TF-IDF (Term Frequency - Inverse Document Frequency) là tích của hai giá trị:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

*   **Tần suất từ (Term Frequency - TF):** Đo lường mức độ thường xuyên của từ $t$ xuất hiện trong tài liệu $d$:
    $$\text{TF}(t, d) = \frac{\text{Số lần từ } t \text{ xuất hiện trong } d}{\text{Tổng số từ trong tài liệu } d}$$
*   **Tần suất nghịch nghịch đảo tài liệu (Inverse Document Frequency - IDF):** Đo lường tầm quan trọng của từ $t$ trên toàn bộ tập tài liệu $D$ (gồm $N$ tài liệu):
    $$\text{IDF}(t, D) = \log \left( \frac{1 + N}{1 + \text{df}(t)} \right) + 1$$
    *(Trong đó $\text{df}(t)$ là số lượng tài liệu chứa từ $t$. Việc cộng 1 ở tử và mẫu để tránh lỗi chia cho 0)*.

*   **Độ tương đồng Cosine (Cosine Similarity):** Dùng để tính toán độ tương đồng giữa hồ sơ người dùng $\vec{U}$ và đặc trưng món ăn $\vec{I}$:
    $$\text{Cosine}(\vec{U}, \vec{I}) = \frac{\vec{U} \cdot \vec{I}}{\|\vec{U}\| \|\vec{I}\|} = \frac{\sum_{i=1}^{M} U_i I_i}{\sqrt{\sum_{i=1}^{M} U_i^2} \sqrt{\sum_{i=1}^{M} I_i^2}}$$

### 3. Phân tích Mã nguồn (Code Trace)
Trong file `src/models.py`, class `ContentBasedRecommender` thực hiện quy trình sau:
```python
# 1. Vector hóa toàn bộ văn bản mô tả món ăn bằng TF-IDF với tối đa 2000 từ đặc trưng nhất
self.vectorizer = TfidfVectorizer(max_features=2000)
self.tfidf_matrix = self.vectorizer.fit_transform(df_foods['combined_features'])

# 2. Xây dựng Hồ sơ người dùng (User Profile) dựa trên các món họ đã đánh giá cao (>= 3.5)
# Lấy trọng số rating trừ đi điểm trung bình để tạo xu hướng thích/ghét
user_ratings = train_df[train_df['userId'] == user_id]
liked_items = user_ratings[user_ratings['rating'] >= 3.5]
# Lấy vector TF-IDF của những món đã thích
liked_tfidf = self.tfidf_matrix[liked_item_indices]
# Tính trung bình cộng các vector này để đại diện cho "gu" của user
user_profile = np.mean(liked_tfidf.toarray(), axis=0)

# 3. Dự đoán rating cho món ăn mới bằng cách nhân Cosine Similarity với 5 (thang điểm tối đa)
similarity = cosine_similarity(user_profile.reshape(1, -1), self.tfidf_matrix[item_idx])
predicted_rating = similarity * 5.0
```

---

## 👥 PHẦN 2: MEMORY-BASED COLLABORATIVE FILTERING (USER-CF & ITEM-CF)

### 1. ELI5 (Giải thích trực quan)
*   **User-CF (Lọc cộng tác dựa trên người dùng):** *"Hãy tìm những người có khẩu vị giống hệt bạn. Nếu họ chấm món Phở Bò 5 sao và bạn chưa bao giờ ăn món đó, hệ thống sẽ gợi ý Phở Bò cho bạn."* (Gợi ý dựa trên bạn bè cùng gu).
*   **Item-CF (Lọc cộng tác dựa trên món ăn):** *"Nếu bạn từng ăn và chấm 5 sao cho Phở Bò, Bún Bò Huế, Hủ Tiếu Nam Vang (các món nước có sợi), hệ thống nhận thấy các món này thường được những người khác chấm điểm tương đồng nhau. Khi đó, nếu bạn chuẩn bị ăn Bún Mọc, hệ thống sẽ gợi ý vì nó rất giống nhóm món ăn bạn đã thích."* (Gợi ý dựa trên sự tương quan thuộc tính tiêu dùng giữa các món).

### 2. Công thức Toán học (The Math)
*   **Độ tương đồng Pearson (Pearson Correlation Coefficient):** Khác với Cosine thông thường, Pearson trừ đi điểm trung bình đánh giá của người dùng để triệt tiêu thiên kiến cá nhân (người khó tính chỉ chấm tối đa 3 sao, người dễ tính luôn chấm 5 sao):
    $$\text{Pearson}(u, v) = \frac{\sum_{i \in I_{uv}} (R_{u,i} - \bar{R}_u)(R_{v,i} - \bar{R}_v)}{\sqrt{\sum_{i \in I_{uv}} (R_{u,i} - \bar{R}_u)^2} \sqrt{\sum_{i \in I_{uv}} (R_{v,i} - \bar{R}_v)^2}}$$
    *(Trong đó $I_{uv}$ là tập các món ăn mà cả user $u$ và $v$ đều đã đánh giá)*.

*   **Công thức dự đoán rating (User-CF):**
    $$\hat{R}_{u,i} = \bar{R}_u + \frac{\sum_{v \in N(u)} \text{Sim}(u, v) \cdot (R_{v,i} - \bar{R}_v)}{\sum_{v \in N(u)} |\text{Sim}(u, v)|}$$
    *(Rating dự đoán bằng điểm trung bình của bạn cộng với trung bình có trọng số độ tương đồng của các hàng xóm xung quanh)*.

### 3. Phân tích Mã nguồn (Code Trace)
Trong class `MemoryCollaborativeFiltering` (`src/models.py`):
```python
# 1. Tính toán ma trận tương đồng (User-User hoặc Item-Item Similarity Matrix)
if self.similarity == 'cosine':
    # Thay thế giá trị khuyết bằng 0 để tính toán
    self.sim_matrix = cosine_similarity(matrix_filled)
elif self.similarity == 'pearson':
    # Sử dụng numpy corrcoef để tính hệ số tương quan Pearson
    self.sim_matrix = np.corrcoef(matrix_filled)

# 2. Dự đoán rating cho User u và Item i (User-CF)
# Lấy k người dùng tương đồng nhất với u đã từng đánh giá món i
sim_users = self.sim_matrix[u_idx]
rated_users_indices = np.where(self.user_item_matrix[:, i_idx] > 0)[0]
# Tìm top-k hàng xóm
top_k_neighbors = sorted(rated_users_indices, key=lambda x: sim_users[x], reverse=True)[:self.k]
# Áp dụng công thức nội suy có trọng số để ra điểm dự đoán
```

---

## 📉 PHẦN 3: FUNK SVD - MATRIX FACTORIZATION (PHÂN RÃ MA TRẬN)

### 1. ELI5 (Giải thích trực quan)
Hãy tưởng tượng ma trận tương tác $101 \times 4000$ khổng lồ giống như một bản đồ kho báu bị mất nhiều mảnh ghép (chỉ có 45.22% ô có điểm). Funk SVD giả định rằng thế giới ẩm thực và con người được vận hành bởi một số lượng nhỏ các **"sở thích ẩn" (gọi là latent factors, ví dụ $f=10$)**:
*   *Món ăn có đặc trưng:* Độ béo, Độ cay, Món nước hay khô, Món ngọt hay mặn...
*   *Người dùng có gu:* Thích béo, Sợ cay, Thích ăn món nước, Thích đồ ngọt...

Funk SVD tách ma trận khổng lồ đó thành 2 ma trận mỏng hơn: Ma trận Người dùng biểu diễn gu ẩn ($101 \times 10$) và Ma trận Món ăn biểu diễn đặc trưng ẩn ($10 \times 4000$). Khi nhân hai ma trận này lại với nhau, chúng ta lấp đầy được tất cả các ô trống bằng các điểm dự đoán cực kỳ chuẩn xác.

```
       Ma trận Tương tác (R)                Ma trận User (P)        Ma trận Item (Q)
        (101 x 4000)                         (101 x 10)              (10 x 4000)
    +-----------------------+            +---------------+       +-----------------------+
    | 5   ?   3   ?   ?   1 |            |  0.2  -0.5 ...|       |  1.1   0.4  -0.2 ...  |
    | ?   2   ?   4   ?   ? |     ≈      |  1.5   0.1 ...|   x   | -0.3   0.9   1.2 ...  |
    | ?   ?   5   ?   ?   5 |            | -0.8   0.7 ...|       +-----------------------+
    +-----------------------+            +---------------+
```

### 2. Công thức Toán học (The Math)
*   **Công thức dự đoán:**
    $$\hat{R}_{u,i} = \mu + b_u + b_i + P_u^T Q_i = \mu + b_u + b_i + \sum_{f=1}^{F} P_{u,f} Q_{i,f}$$
    *(Trong đó $\mu$ là trung bình toàn bộ rating; $b_u, b_i$ là bias của user/item; $P_u$ là vector gu ẩn của user; $Q_i$ là vector đặc trưng ẩn của item)*.

*   **Hàm mất mát (Loss Function) cần tối thiểu hóa:**
    $$\min_{b, P, Q} \sum_{(u,i) \in R_{train}} \left( R_{u,i} - \hat{R}_{u,i} \right)^2 + \lambda \left( b_u^2 + b_i^2 + \|P_u\|_2^2 + \|Q_i\|_2^2 \right)$$
    *(Vế sau là L2 Regularization với tham số phạt $\lambda$, giúp mô hình không bị quá khớp - overfitting)*.

*   **Thuật toán cập nhật tham số (Stochastic Gradient Descent - SGD):**
    Với mỗi tương tác thực tế $(u, i)$ có sai số $e_{u,i} = R_{u,i} - \hat{R}_{u,i}$:
    $$b_u \leftarrow b_u + \gamma (e_{u,i} - \lambda b_u)$$
    $$b_i \leftarrow b_i + \gamma (e_{u,i} - \lambda b_i)$$
    $$P_{u,f} \leftarrow P_{u,f} + \gamma (e_{u,i} Q_{i,f} - \lambda P_{u,f})$$
    $$Q_{i,f} \leftarrow Q_{i,f} + \gamma (e_{u,i} P_{u,f} - \lambda Q_{i,f})$$
    *(Với $\gamma$ là tốc độ học - learning rate)*.

### 3. Phân tích Mã nguồn (Code Trace)
Trong class `FunkSVDRecommender` (`src/models.py`):
```python
# 1. Khóa seed ngẫu nhiên để đảm bảo kết quả chạy lại luôn giống nhau 100%
np.random.seed(42)
self.P = np.random.normal(0, 0.1, (num_users, self.n_factors))
self.Q = np.random.normal(0, 0.1, (num_items, self.n_factors))

# 2. Vòng lặp tối ưu hóa SGD qua từng Epoch
for epoch in range(self.epochs):
    indices = np.random.permutation(len(ratings)) # Trộn ngẫu nhiên dữ liệu học
    for idx in indices:
        u, i, r = u_indices[idx], i_indices[idx], ratings[idx]
        # Điểm dự đoán hiện tại
        pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        err = r - pred # Tính sai số
        
        # Cập nhật các trọng số theo Gradient giảm dần
        self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
        self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
        
        # Lưu trữ tạm thời vector cũ để tránh cập nhật chéo trong cùng bước
        P_u_old = self.P[u].copy()
        self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
        self.Q[i] += self.lr * (err * P_u_old - self.reg * self.Q[i])
```

---

## 🧠 PHẦN 4: NEURAL COLLABORATIVE FILTERING (NCF) - HỌC SÂU

### 1. ELI5 (Giải thích trực quan)
Funk SVD ở trên rất thông minh nhưng nó có một nhược điểm chí tử: nó nhân vô hướng (dot-product) giữa hai ma trận ẩn. Phép nhân vô hướng là phép toán tuyến tính, tức là nó chỉ cộng nhân đơn thuần, không học được các quan hệ phức tạp, uốn lượn hay phi tuyến tính.

Neural CF giải quyết vấn đề này bằng cách đưa gu người dùng và đặc trưng món ăn vào một **Mạng Thần Kinh Nhân Tạo (Multi-Layer Perceptron - MLP)**. Giống như bộ não con người, mạng nơ-ron này có nhiều lớp neuron liên kết với nhau, có khả năng học các mối quan hệ logic phức tạp, ví dụ: *"Nếu user thích ngọt VÀ thích món nước THÌ họ sẽ cực kỳ ghét món cay khô, nhưng nếu là buổi tối lạnh thì họ lại sẵn sàng thử món cay nước"*. Những logic phức tạp này, Funk SVD tuyến tính hoàn toàn bất lực.

```
    User ID ---> [ User Embedding ] \
                                     ===> [ Lớp Ghép Sát - Concatenate ] ===> Lớp MLP 1 ===> Lớp MLP 2 ===> Rating Dự Đoán [0, 5]
    Item ID ---> [ Item Embedding ] /
```

### 2. Công thức Toán học (The Math)
*   **Tầng Embedding:** Ánh xạ các chỉ mục thô (ID) sang không gian vector liên tục:
    $$\mathbf{v}_u^U = \mathbf{P} \cdot \mathbf{x}_u, \quad \mathbf{v}_i^I = \mathbf{Q} \cdot \mathbf{x}_i$$
*   **Tầng kết hợp (Concatenation):**
    $$\mathbf{a}_0 = [\mathbf{v}_u^U, \mathbf{v}_i^I]$$
*   **Các tầng ẩn MLP (Multi-Layer Perceptron):**
    $$\mathbf{a}_1 = a(\mathbf{W}_1 \mathbf{a}_0 + \mathbf{b}_1)$$
    $$\mathbf{a}_2 = a(\mathbf{W}_2 \mathbf{a}_1 + \mathbf{b}_2)$$
    *(Với $a(x) = \max(0, x)$ là hàm kích hoạt phi tuyến ReLU)*.
*   **Tầng đầu ra (Output Layer):** Dự đoán rating rating bằng hàm kích hoạt tuyến tính hoặc Sigmoid tỷ lệ hóa:
    $$\hat{R}_{u,i} = \mathbf{w}_{out}^T \mathbf{a}_L$$

### 3. Phân tích Mã nguồn (Code Trace)
Trong class `NCFNet` và `TorchNCFRecommender` (`src/models.py`):
```python
# 1. Định nghĩa cấu trúc mạng Neural trong PyTorch
class NCFNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        
        # Thiết lập các tầng MLP
        mlp_modules = []
        input_dim = embedding_dim * 2 # Đầu vào là concatenate của 2 vector embedding
        for output_dim in layers:
            mlp_modules.append(nn.Linear(input_dim, output_dim))
            mlp_modules.append(nn.ReLU()) # Hàm phi tuyến tính ReLU
            input_dim = output_dim
        self.mlp = nn.Sequential(*mlp_modules)
        self.prediction_layer = nn.Linear(layers[-1], 1) # Đầu ra dự đoán điểm số 1 chiều

    def forward(self, user_indices, item_indices):
        user_vector = self.user_embed(user_indices)
        item_vector = self.item_embed(item_indices)
        x = torch.cat([user_vector, item_vector], dim=-1) # Concatenate
        x = self.mlp(x)
        return self.prediction_layer(x).squeeze()
```

---

## 📈 PHẦN 5: BÍ ẨN 8 ĐỘ ĐO ĐÁNH GIÁ (THE 8 METRICS)

Hệ thống đánh giá chia làm 2 chiến tuyến rõ rệt: **Dự đoán điểm (Regression)** và **Xếp hạng danh sách (Ranking)**.

### A. Nhóm Regression (Độ chính xác trị số)

#### 1. MSE (Mean Squared Error - Sai số bình phương trung bình)
*   **ELI5:** Đo xem trung bình các phán đoán của bạn lệch bao nhiêu điểm, rồi bình phương sai số đó lên để phạt thật nặng những lỗi đoán lệch quá xa (ví dụ thực tế là 5 sao mà bạn đoán 1 sao).
*   **Toán học:** 
    $$\text{MSE} = \frac{1}{N} \sum_{k=1}^N (y_k - \hat{y}_k)^2$$

#### 2. RMSE (Root Mean Squared Error - Căn bậc hai sai số bình phương trung bình)
*   **ELI5:** Lấy căn bậc hai của MSE để đưa đơn vị sai số quay trở về đúng thang điểm gốc $[0, 5]$ giúp con người dễ hình dung. Ví dụ: RMSE = 1.5 nghĩa là trung bình mô hình đoán lệch $\pm 1.5$ sao.
*   **Toán học:** 
    $$\text{RMSE} = \sqrt{\text{MSE}}$$

#### 3. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)
*   **ELI5:** Chỉ đơn giản là cộng tất cả các khoảng cách sai số (không bình phương) rồi chia trung bình. Độ đo này thân thiện hơn, không phạt quá nặng các lỗi cực đoan như RMSE.
*   **Toán học:** 
    $$\text{MAE} = \frac{1}{N} \sum_{k=1}^N |y_k - \hat{y}_k|$$

#### 4. NMAE (Normalized Mean Absolute Error - MAE chuẩn hóa)
*   **ELI5:** Lấy MAE chia cho khoảng thang điểm tối đa (trong trường hợp của chúng ta là $5 - 0 = 5$) để đưa sai số về tỉ lệ phần trăm từ $0$ đến $1$, giúp so sánh giữa các bộ dữ liệu khác nhau.
*   **Toán học:** 
    $$\text{NMAE} = \frac{\text{MAE}}{R_{max} - R_{min}} = \frac{\text{MAE}}{5}$$

---

### B. Nhóm Ranking (Độ chính xác của danh sách Top-10)
*Để đánh giá Ranking, một món ăn được coi là "đáng xem/đáng ăn" nếu điểm tương tác thực tế của người dùng dành cho nó $\ge 3.5$ (ngưỡng Relevance).*

#### 5. Precision@10 (Độ chính xác của danh sách Top-10)
*   **ELI5:** Hệ thống gợi ý cho bạn 10 món. Bạn ăn thử cả 10 món đó. Nếu bạn thích 4 món (chấm $\ge 3.5$ sao), thì Precision@10 là $4/10 = 40.0\%$.
*   **Toán học:** 
    $$\text{Precision@K} = \frac{|\{\text{Món trong Top-K có } R_{real} \ge 3.5\}|}{K}$$

#### 6. Recall@10 (Độ bao phủ của danh sách Top-10)
*   **ELI5:** Thực tế trong tủ lạnh có tổng cộng 50 món ăn bạn cực kỳ thích. Hệ thống chỉ tìm ra được và gợi ý cho bạn 5 món thích trong danh sách Top-10 đó. Recall@10 của hệ thống là $5/50 = 10.0\%$.
*   **Toán học:** 
    $$\text{Recall@K} = \frac{|\{\text{Món trong Top-K có } R_{real} \ge 3.5\}|}{|\{\text{Tất cả món thực tế user thích trong tập Test}\}|}$$

#### 7. NDCG@10 (Normalized Discounted Cumulative Gain - Đánh giá thứ hạng có chiết khấu)
*   **ELI5:** Không chỉ quan tâm bạn thích bao nhiêu món, NDCG quan tâm đến **thứ tự gợi ý**. Nếu món bạn thích nằm ở vị trí số 1, bạn sẽ vui hơn nhiều so với việc nó bị xếp ở vị trí số 10. Những món ở vị trí dưới sẽ bị "phạt chia log" (chiết khấu) để giảm giá trị đóng góp vào độ hài lòng chung.
*   **Toán học:** 
    $$\text{DCG@K} = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
    *(IDCG@K là giá trị DCG lý tưởng khi danh sách được sắp xếp hoàn hảo từ thích nhất đến ít thích nhất)*.

#### 8. MRR (Mean Reciprocal Rank - Thứ hạng nghịch đảo trung bình)
*   **ELI5:** Hệ thống chỉ quan tâm xem **món đầu tiên bạn thích** nằm ở vị trí thứ mấy trong danh sách gợi ý. Nếu món đầu tiên bạn thích nằm ở vị trí số 1, điểm là $1/1 = 1.0$. Nếu nó nằm ở vị trí thứ 5, điểm là $1/5 = 0.2$. Càng xếp món chuẩn lên đầu, MRR càng cao.
*   **Toán học:** 
    $$\text{MRR} = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{\text{rank}_u^*}$$
    *(Với $\text{rank}_u^*$ là vị trí của món ăn liên quan đầu tiên được tìm thấy của người dùng $u$)*.

---

## 📈 PHẦN 6: "SCIENCE BEHIND THE PLOT TWIST" (TẠI SAO NCF THẮNG RMSE NHƯNG USER-CF THẮNG RANKING?)

Đây là phần biện luận học thuật quan trọng nhất trong báo cáo khóa luận của bạn. Tại sao mạng học sâu Neural CF (NCF) tối tân, học phi tuyến rất đỉnh, đạt sai số dự đoán điểm số (RMSE) thấp nhất nhưng khi xếp hạng gợi ý món ăn (Top-K Ranking) lại suýt soát hoặc thỉnh thoảng thua thuật toán User-CF cổ điển?

### 1. Phân tích nguyên nhân sâu xa (Học thuật cao cấp)
Hiện tượng này được gọi là **Sự bất đối xứng giữa Hàm tối ưu hóa và Mục tiêu tiêu dùng (Optimization-Evaluation Mismatch)**.

*   **NCF được huấn luyện bằng hàm loss MSE:** 
    Mạng NCF được thiết kế để ép sai số bình phương giữa điểm dự đoán $\hat{y}$ và điểm thực tế $y$ nhỏ nhất có thể. Do đó, NCF tập trung học rất tốt các giá trị điểm trung bình (ví dụ: dự đoán chính xác món này là 3.6 sao thay vì 4.2 sao). Nó làm giảm RMSE cực kỳ hiệu quả.
*   **Nhưng Ranking không cần giá trị tuyệt đối chính xác:** 
    Để gợi ý danh sách Top-10 tốt nhất, hệ thống chỉ cần biết **quan hệ thứ tự lớn hơn/nhỏ hơn (relative order)** giữa các món ăn. Ví dụ: chỉ cần mô hình biết món A tốt hơn món B (xếp A lên trước B) là điểm Precision@10 và NDCG đã đạt tuyệt đối, bất kể mô hình đoán món A là 4.9 sao (thực tế 4.5) hay đoán món A là 3.8 sao (thực tế 4.0).
*   **Hiệu ứng làm mịn của mạng Nơ-ron (Smoothing effect):**
    NCF học các biểu diễn vector (embeddings) liên tục thông qua các tầng MLP tuyến tính kết hợp phi tuyến ReLU. Quá trình này có xu hướng "làm mịn" các dự đoán rating về gần điểm trung bình ($\approx 3.7$), khiến mô hình mất đi khả năng phân biệt ranh giới rõ rệt giữa các món rất xuất sắc và các món khá tốt.
*   **Ưu thế thống kê của User-CF Cosine trên dữ liệu tương tác MXH dày:**
    Bộ dữ liệu của bạn có **Density cực kỳ lớn (45.22%)**. Trong một không gian tương tác dày đặc như vậy, các "hàng xóm" của người dùng lộ diện rất rõ ràng và chuẩn xác. User-CF chỉ cần sao chép trực tiếp hành vi thực tế của những người hàng xóm này để đẩy các món được thích nhất lên đầu danh sách, không cần thông qua các bộ lọc nén hay tầng học sâu trung gian làm mất mát tín hiệu phân bậc gốc.

---

## 🛠 PHẦN 7: PHÂN TÍCH LỖI ĐA CHIỀU (ERROR ANALYSIS) VÀ HƯỚNG KHẮC PHỤC

Để luận văn đạt điểm tối đa từ hội đồng phản biện, phần phân tích lỗi phải chỉ rõ các lỗ hổng kỹ thuật thực tế và đề xuất giải pháp khoa học:

### 1. Vấn đề Khởi đầu lạnh (Cold-Start Problem)
*   **Biểu hiện trong Code:** Khi gặp User mới chưa tương tác hoặc Item mới tinh, Funk SVD và NCF không có dữ liệu lịch sử để cập nhật embedding. Trọng số embedding của họ sẽ giữ nguyên giá trị khởi tạo ngẫu nhiên hoặc hội tụ về điểm trung bình toàn cục $\mu$ ($\approx 2.5$).
*   **Giải pháp nâng cao:** Triển khai cơ chế **Khởi tạo Embedding dựa trên Metadata (Content-to-Collaborative Warm Up)**. Sử dụng mô hình CBF (TF-IDF) để sinh ra vector đặc trưng văn bản của Item mới, sau đó đi qua một mạng chuyển đổi tuyến tính (Projection Layer) để ánh xạ vector này trực tiếp vào không gian Latent Factor của Funk SVD hoặc NCF làm embedding khởi tạo ban đầu.

### 2. Thiên kiến Phổ biến (Popularity Bias)
*   **Biểu hiện trong Code:** Các món ăn như "Phở Bò", "Cà Phê Sữa Đá" có hàng ngàn lượt đánh giá cao trong dataset. Hệ thống có xu hướng gợi ý các món này cho mọi người dùng vì chúng mang lại điểm an toàn cao, dẫn đến danh sách gợi ý nhàm chán, thiếu tính cá nhân hóa (Personalization).
*   **Giải pháp nâng cao:** Áp dụng thuật toán **Chiết khấu độ phổ biến (Popularity Penalty) ở bước Re-ranking**:
    $$R_{\text{final}}(u, i) = R_{\text{predict}}(u, i) - \beta \cdot \log(\text{Popularity}(i))$$
    *(Trong đó $\beta$ là hệ số điều chỉnh mức phạt món ăn phổ biến)*.

### 3. Ma trận dẹt và sự bất ổn định của Item-CF
*   **Biểu hiện trong Code:** Số lượng người dùng quá ít ($101$) so với số lượng món ăn ($4000$). Ma trận tương tác có hình dáng rất dẹt. Khi tính tương quan giữa các món ăn (Item-CF), hai món ăn bất kỳ rất khó tìm được tập người dùng chung đủ lớn để tính độ tương đồng tin cậy, làm Item-CF có kết quả kém ổn định.
*   **Giải pháp nâng cao:** Chuyển sang sử dụng các mô hình nén chiều không gian như Funk SVD hoặc ưu tiên sử dụng User-CF vì số lượng User nhỏ ($101$) giúp tính toán ma trận tương đồng người dùng cực kỳ nhanh và chính xác.

### 4. Nhiễu văn bản (Text Noise) trong CBF
*   **Biểu hiện trong Code:** CBF sử dụng TF-IDF thô trên các từ tiếng Việt không dấu hoặc ghép từ chưa chuẩn (ví dụ "thịt bò", "bò xào" bị tách thành các từ đơn độc lập), làm giảm độ chính xác ngữ nghĩa, khiến CBF có RMSE cao nhất ($1.9502$).
*   **Giải pháp nâng cao:** Thay thế bộ vector hóa TF-IDF bằng các mô hình ngôn ngữ lớn chuyên biệt cho tiếng Việt như **PhoBERT** (Pre-trained BERT for Vietnamese) để trích xuất ngữ nghĩa (semantic embeddings) của món ăn trước khi tính tương đồng Cosine.

---

## 🎓 TỔNG KẾT BÀI HỌC CHO KHÓA LUẬN
1.  **Học sâu không phải là vạn năng:** NCF rất tốt cho việc khớp điểm số, nhưng các phương pháp Heuristic (như User-CF) vẫn cực kỳ mạnh mẽ cho việc xếp hạng thực tế nhờ không bị suy hao tín hiệu qua các lớp ẩn.
2.  **Đánh giá đa chiều:** Khi viết luận văn, luôn luôn phải so sánh cả Regression và Ranking. Một hệ thống gợi ý thương mại điện tử thực tế luôn ưu tiên độ chính xác Ranking hơn là điểm dự đoán tuyệt đối.
3.  **Tái lập kết quả:** Đảm bảo cố định seed ngẫu nhiên (`random_state=42`) cho tất cả các thư viện (Numpy, PyTorch, Scikit-learn) là điều kiện bắt buộc để bài báo khoa học được công nhận.

