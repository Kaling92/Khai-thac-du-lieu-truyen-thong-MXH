# Đồ án: Hệ Thống Khuyến Nghị Món Ăn Việt Nam (ViFoodRec)

Đồ án môn **Khai thác dữ liệu truyền thông xã hội** — Đề tài 01: Xây dựng dữ liệu và đánh giá các mô hình khuyến nghị món ăn.

## Cấu trúc dự án

```
├── app.py                          # Ứng dụng demo Streamlit
├── src/
│   ├── preprocessing.py            # Tiền xử lý dữ liệu
│   ├── models.py                   # 4 nhóm mô hình khuyến nghị
│   ├── evaluate.py                 # Độ đo đánh giá
│   ├── train.py                    # Huấn luyện & sinh biểu đồ
│   └── update_latex.py             # Cập nhật số liệu vào báo cáo
├── report/
│   ├── report.tex                  # Báo cáo LaTeX (mẫu ACL 2 cột)
│   ├── custom.bib                  # Tài liệu tham khảo
│   ├── acl.sty, acl_natbib.bst     # Style ACL
│   └── figures/                    # Biểu đồ kết quả (sau khi train)
├── slides/
│   ├── generate_slides.py          # Script tạo slide PPTX
│   └── slide_research.pptx         # Slide mới (RQ, Research Gap, Discussion)
└── Clean Dataset-.../Clean Dataset/
    ├── foods.csv                   # 4.000 món ăn
    └── ratings.csv                 # 182.678 lượt đánh giá
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy thí nghiệm

```bash
# 1. Huấn luyện tất cả mô hình và sinh biểu đồ
python src/train_light.py

# 2. Cập nhật số liệu vào báo cáo LaTeX
python src/update_latex.py

# 3. Tạo file thuyết trình
python slides/generate_slides.py

# 4. Chạy ứng dụng demo
streamlit run app.py
```

## Báo cáo PDF (Overleaf)

1. Tạo project mới trên [Overleaf](https://www.overleaf.com/)
2. Upload toàn bộ thư mục `report/` (bao gồm `figures/`)
3. Đổi Compiler sang **XeLaTeX**
4. Biên dịch và tải PDF

## Bốn phương pháp triển khai

| # | Phương pháp | Mô tả |
|---|-------------|-------|
| 1 | Content-Based (CBF) | TF-IDF + Cosine similarity trên metadata món ăn |
| 2 | Memory CF | User-CF và Item-CF (Cosine, Pearson) |
| 3 | Funk SVD | Phân rã ma trận bằng SGD |
| 4 | Neural CF (NCF) | Mạng nơ-ron PyTorch / MLP scikit-learn |

## Độ đo đánh giá

- **Regression**: MSE, RMSE, MAE, NMAE
- **Ranking**: Precision@10, Recall@10, NDCG@10, MRR
