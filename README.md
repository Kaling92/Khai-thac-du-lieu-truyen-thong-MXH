# ViFoodRec - Hệ Thống Khuyến Nghị Món Ăn Việt Nam

Dự án môn học **Khai thác dữ liệu truyền thông xã hội**. Hệ thống khuyến nghị đồ ăn cá nhân hóa với sự kết hợp giữa Recommendation Systems truyền thống và Natural Language Processing (SBERT).

## Tính Năng Cốt Lõi
Hệ thống hiện tại là sự hợp nhất của 2 pipeline (Train và SBERT) thành một giao diện duy nhất:
1. **Tổng Quan Dữ Liệu (EDA)**: Phân tích phân phối rating, số lượng user, và các thuộc tính món ăn.
2. **Khuyến Nghị CBF & Collaborative Filtering**: 
   - Content-Based Filtering (CBF)
   - User-based & Item-based CF
   - Funk SVD (Matrix Factorization)
   - Neural Collaborative Filtering (NCF)
3. **Semantic Search & Hybrid BERT**: Gợi ý kết hợp hành vi người dùng (NCF) và nội dung ngữ nghĩa văn bản món ăn (SBERT).
4. **Compare Models**: So sánh trực quan RMSE, MAE, Recall@K, ... của tất cả các mô hình.

## Kiến Trúc Project
```text
project/
├── data/
│     ├── raw/                 # Clean Dataset/foods.csv, ratings.csv
│     └── processed/           # processed_data/
├── models/                    # ncf/, bert/, hybrid_bert/
├── src/
│     ├── config.py            # Chứa CONSTANTS và Paths tuyệt đối
│     ├── data_loader.py       # Caching dataset với @st.cache_data
│     ├── preprocessing.py     
│     ├── models.py            
│     └── evaluate.py          
├── pipelines/
│     ├── train_pipeline.py    
│     ├── sbert_pipeline.py    
│     └── reporting.py         
├── app/
│     ├── main.py              # Entry point duy nhất
│     └── pages/               # Multi-page Streamlit
└── docs/                      # report/, educational_manual.md
```

## Hướng Dẫn Sử Dụng
Khởi động giao diện Dashboard:
```bash
1/ Run 
 .\.venv\Scripts\Activate

2/ 
pip install -r requirements.txt

3/
streamlit run app/main.py
```
Mở đường dẫn `http://localhost:8501` trên trình duyệt.

Chạy huấn luyện (CBF, CF, SVD, NCF):
python pipelines/train_pipeline.py
Chạy phần nhúng ngôn ngữ (SBERT):
python pipelines/sbert_embedding1.py (Dữ liệu đầu vào sẽ tự động được lấy theo chuẩn đã cấu hình trong src/config.py).

