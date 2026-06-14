"""Generate academic diagrams and EDA figures for the ViFoodRec report."""
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette
C_NAVY = "#1E3A8A"
C_BLUE = "#3B82F6"
C_ORANGE = "#F5A623"
C_GREEN = "#10B981"
C_PURPLE = "#8B5CF6"
C_RED = "#EF4444"
C_GRAY = "#6B7280"
C_LIGHT = "#EFF6FF"


def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


def draw_box(ax, x, y, w, h, text, color=C_BLUE, textcolor="white", fontsize=9):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold", wrap=True, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.8))


def fig_cbf_recommendation_process():
    """Figure 5 style: 4-step CBF recommendation process from sample paper."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Quy trình Khuyến nghị bằng Content-Based Filtering (theo Tran et al., 2024)",
                 fontsize=12, fontweight="bold", color=C_NAVY, pad=12)

    draw_box(ax, 0.3, 3.5, 2.0, 1.0, "(1) Input Data\nfoods_modeling", C_ORANGE, "white", 9)
    draw_box(ax, 2.8, 3.5, 2.4, 1.0, "(2) Trích xuất thuộc tính\n& gán trọng số", C_BLUE, "white", 8)
    draw_box(ax, 5.7, 3.8, 2.0, 0.7, "description\n(0.25)", C_PURPLE, "white", 7)
    draw_box(ax, 5.7, 2.9, 2.0, 0.7, "ingredients\n(0.60)", C_PURPLE, "white", 7)
    draw_box(ax, 8.2, 3.8, 1.8, 0.7, "dish_tags\n(0.10)", C_GRAY, "white", 7)
    draw_box(ax, 8.2, 2.9, 1.8, 0.7, "nutrient\n(0.05)", C_GRAY, "white", 7)

    draw_box(ax, 0.5, 1.5, 4.5, 1.2,
             "(3) Tính độ tương đồng\n(Cosine / TF-IDF)\n→ Danh sách món liên quan", C_NAVY, "white", 9)
    draw_box(ax, 5.5, 1.5, 5.0, 1.2,
             "(4) Lọc theo yêu cầu\n(calories, cooking_time, dish_type, serving_size)\n→ Top-K gợi ý cuối cùng", C_GREEN, "white", 9)

    draw_arrow(ax, 2.3, 4.0, 2.8, 4.0)
    draw_arrow(ax, 5.2, 4.0, 5.7, 3.5)
    draw_arrow(ax, 6.7, 2.9, 2.5, 2.7)
    draw_arrow(ax, 5.0, 2.1, 5.5, 2.1)
    _save(fig, "cbf_recommendation_process.png")


def fig_food_recommendation_system():
    """Figure 6 style: Food Recommendation System Input/Output."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Hệ thống Khuyến nghị Món ăn Việt Nam (Food Recommendation System)",
                 fontsize=12, fontweight="bold", color=C_NAVY, pad=12)

    # Left: Content & Item Based
    ax.text(2.5, 5.5, "CONTENT & ITEM BASED", ha="center", fontsize=10, fontweight="bold", color=C_NAVY)
    draw_box(ax, 0.3, 4.5, 1.2, 0.7, "Input", C_ORANGE, "white", 9)
    draw_box(ax, 1.8, 4.2, 3.5, 1.3, "REQUIREMENTS\nfood type | serving size\ncooking time | calories", C_BLUE, "white", 8)
    draw_box(ax, 0.5, 2.8, 2.0, 0.9, "Food\nSelection", C_PURPLE, "white", 9)
    draw_box(ax, 2.8, 2.8, 2.5, 0.9, "CBF / Item-CF\nEngine", C_NAVY, "white", 9)
    draw_box(ax, 0.5, 1.2, 4.8, 0.9, "RELATED FOOD LIST (Output)", C_GREEN, "white", 10)

    # Right: User Based
    ax.text(8.5, 5.5, "USER BASED (CF / SVD / NCF)", ha="center", fontsize=10, fontweight="bold", color=C_NAVY)
    draw_box(ax, 5.8, 4.5, 1.2, 0.7, "Input", C_ORANGE, "white", 9)
    draw_box(ax, 7.3, 4.2, 3.5, 1.3, "REQUIREMENTS\nliked foods | user ID\nmodel selection", C_BLUE, "white", 8)
    draw_box(ax, 6.0, 2.8, 2.2, 0.9, "User History\n(Ratings)", C_PURPLE, "white", 8)
    draw_box(ax, 8.5, 2.8, 2.3, 0.9, "User-CF / SVD / NCF", C_NAVY, "white", 8)
    draw_box(ax, 5.8, 1.2, 5.0, 0.9, "PERSONALIZED FOOD LIST (Output)", C_GREEN, "white", 10)

    ax.plot([5.3, 5.3], [1, 5.2], color=C_GRAY, linestyle="--", lw=1.5)
    _save(fig, "food_recommendation_system.png")


def fig_ratings_before_after():
    """Figure 4 style: ratings count before/after filling missing values."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["Trước điền\n(~50.000)", "Sau điền\n(~182.678)"]
    values = [50000, 182678]
    colors = [C_ORANGE, C_GREEN]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Số lượng Ratings")
    ax.set_title("Thống kê số Ratings trước và sau điền giá trị thiếu\n(theo quy trình ViFoodRec gốc)", fontweight="bold", color=C_NAVY)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3000,
                f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 210000)
    plt.tight_layout()
    _save(fig, "ratings_before_after.png")


def fig_textual_eda():
    """Figure 2 style: textual attributes sample visualization."""
    foods_path = os.path.join(PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "Clean Dataset", "foods.csv")
    df = pd.read_csv(foods_path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle("Phân tích trực quan một số thuộc tính văn bản (Textual Attributes)", fontweight="bold", color=C_NAVY)
    for ax, col, title in zip(axes, ["serving_size", "cooking_time", "calories"],
                              ["Serving Size", "Cooking Time (phút)", "Calories (kcal)"]):
        if col == "serving_size":
            counts = df[col].value_counts().head(6)
            ax.bar(range(len(counts)), counts.values, color=C_BLUE)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([str(x)[:12] for x in counts.index], rotation=30, ha="right", fontsize=7)
        else:
            ax.hist(df[col].dropna(), bins=20, color=C_ORANGE, edgecolor="white")
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    _save(fig, "textual_numerical_eda.png")


def fig_data_collection():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Quy trình Thu thập Dữ liệu ViFoodRec từ Mạng Xã hội Ẩm thực",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=15)

    sources = [
        (0.3, 4.5, "Monngonmoingay.com\n(Công thức & hình ảnh)"),
        (0.3, 3.0, "Cooky.vn\n(Cộng đồng nấu ăn)"),
        (0.3, 1.5, "Diễn đàn MXH\n(Bình luận & đánh giá)"),
    ]
    for x, y, t in sources:
        draw_box(ax, x, y, 2.2, 1.0, t, C_ORANGE, "white", 8)

    draw_box(ax, 3.5, 2.5, 2.0, 1.5, "Web Crawling\n& Scraping", C_NAVY, "white", 9)
    for y in [5.0, 3.5, 2.0]:
        draw_arrow(ax, 2.5, y, 3.5, 3.25)

    draw_box(ax, 6.2, 2.5, 1.8, 1.5, "Làm sạch\n& Chuẩn hóa", C_BLUE, "white", 9)
    draw_arrow(ax, 5.5, 3.25, 6.2, 3.25)

    draw_box(ax, 8.5, 3.3, 1.3, 0.9, "foods.csv\n4.000 món", C_GREEN, "white", 8)
    draw_box(ax, 8.5, 2.0, 1.3, 0.9, "ratings.csv\n182K ratings", C_GREEN, "white", 8)
    draw_arrow(ax, 8.0, 3.5, 8.5, 3.75)
    draw_arrow(ax, 8.0, 3.0, 8.5, 2.45)

    ax.text(5, 0.5, "Nguồn: Tran et al. (2024) — ViFoodRec Dataset",
            ha="center", fontsize=8, color=C_GRAY, style="italic")
    _save(fig, "data_collection_process.png")


def fig_preprocessing_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("Quy trình Tiền xử lý Dữ liệu (Data Preprocessing Pipeline)",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=12)

    steps = [
        (0.2, 1.5, "Raw Data\n(foods + ratings)", C_ORANGE),
        (2.3, 1.5, "Unicode NFC\nLowercase", C_BLUE),
        (4.4, 1.5, "Loại bỏ\nký tự đặc biệt", C_BLUE),
        (6.5, 1.5, "Ghép đặc trưng\nvăn bản (CBF)", C_PURPLE),
        (8.6, 1.5, "Chia Train/Test\n90% / 10%", C_GREEN),
    ]
    for i, (x, y, t, c) in enumerate(steps):
        draw_box(ax, x, y, 1.8, 1.2, t, c, "white", 8)
        if i < len(steps) - 1:
            draw_arrow(ax, x + 1.8, y + 0.6, steps[i + 1][0], y + 0.6)

    draw_box(ax, 2.3, 0.2, 1.8, 0.8, "Chuẩn hóa số\n(calories, protein...)", C_GRAY, "white", 7)
    draw_box(ax, 6.5, 0.2, 1.8, 0.8, "TF-IDF Vector\n(2.000 features)", C_GRAY, "white", 7)
    _save(fig, "preprocessing_pipeline.png")


def fig_research_framework():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Khung Nghiên cứu Tổng thể (Research Framework)",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=12)

    draw_box(ax, 3.5, 6.0, 3.0, 0.7, "BÀI TOÁN: Khuyến nghị Món ăn Việt Nam", C_NAVY, "white", 10)
    draw_box(ax, 0.5, 4.5, 2.0, 0.9, "Bộ dữ liệu\nViFoodRec", C_ORANGE, "white", 9)
    draw_box(ax, 3.0, 4.5, 4.0, 0.9, "4 Phương pháp Khuyến nghị", C_BLUE, "white", 10)
    draw_box(ax, 7.5, 4.5, 2.0, 0.9, "Đánh giá\n(MSE, NDCG...)", C_GREEN, "white", 9)

    methods = ["CBF\n(TF-IDF)", "User/Item-CF", "Funk SVD", "NCF\n(Deep Learning)"]
    for i, m in enumerate(methods):
        draw_box(ax, 3.0 + i * 1.0, 3.2, 0.9, 0.9, m, C_PURPLE, "white", 7)

    draw_box(ax, 1.5, 1.5, 2.5, 0.9, "Phân tích Lỗi\n(Cold-start, Bias...)", C_RED, "white", 9)
    draw_box(ax, 4.5, 1.5, 2.5, 0.9, "Lựa chọn Mô hình\nTối ưu", C_NAVY, "white", 9)
    draw_box(ax, 7.5, 1.5, 2.0, 0.9, "Hệ thống\nDemo Streamlit", C_ORANGE, "white", 9)

    draw_arrow(ax, 5.0, 6.0, 1.5, 5.4)
    draw_arrow(ax, 5.0, 6.0, 5.0, 5.4)
    draw_arrow(ax, 5.0, 6.0, 8.5, 5.4)
    draw_arrow(ax, 1.5, 4.5, 1.5, 2.4)
    draw_arrow(ax, 5.0, 4.5, 5.0, 3.2)
    draw_arrow(ax, 8.5, 4.5, 5.75, 2.4)
    draw_arrow(ax, 5.75, 1.5, 8.5, 1.95)
    _save(fig, "research_framework.png")


def fig_method_architecture(name, filename, layers):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(f"Kiến trúc {name}", fontsize=12, fontweight="bold", color=C_NAVY, pad=10)

    colors = [C_ORANGE, C_BLUE, C_PURPLE, C_GREEN, C_NAVY]
    n = len(layers)
    total_w = n * 1.4 + (n - 1) * 0.5
    start_x = (9 - total_w) / 2

    for i, (label, sub) in enumerate(layers):
        x = start_x + i * 1.9
        draw_box(ax, x, 1.5, 1.4, 1.3, f"{label}\n{sub}", colors[i % len(colors)], "white", 8)
        if i < n - 1:
            draw_arrow(ax, x + 1.4, 2.15, x + 1.9, 2.15)
    _save(fig, filename)


def fig_recommendation_process():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Quy trình Khuyến nghị Món ăn (Recommendation Process)",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=12)

    steps = [
        "Người dùng\n(User ID)",
        "Lịch sử\nRatings",
        "Chọn\nMô hình",
        "Dự đoán\nRating",
        "Xếp hạng\nTop-K",
        "Lọc\n(Dcalo, thời gian)",
        "Danh sách\nGợi ý",
    ]
    for i, s in enumerate(steps):
        x = 0.3 + i * 1.5
        c = C_ORANGE if i == 0 else (C_GREEN if i == len(steps) - 1 else C_BLUE)
        draw_box(ax, x, 2.0, 1.3, 1.2, s, c, "white", 8)
        if i < len(steps) - 1:
            draw_arrow(ax, x + 1.3, 2.6, x + 1.5, 2.6)

    draw_box(ax, 3.5, 0.3, 4.0, 0.9, "4 Models: CBF | User-CF | Item-CF | SVD | NCF", C_PURPLE, "white", 8)
    draw_arrow(ax, 5.0, 2.0, 5.0, 1.2)
    _save(fig, "recommendation_process.png")


def fig_system_architecture():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Kiến trúc Hệ thống Khuyến nghị Món ăn Việt Nam",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=12)

    # Layer labels
    ax.text(0.2, 6.5, "Presentation Layer", fontsize=9, color=C_GRAY, fontweight="bold")
    ax.text(0.2, 5.0, "Application Layer", fontsize=9, color=C_GRAY, fontweight="bold")
    ax.text(0.2, 3.2, "Recommendation Engine", fontsize=9, color=C_GRAY, fontweight="bold")
    ax.text(0.2, 1.2, "Data Layer", fontsize=9, color=C_GRAY, fontweight="bold")

    draw_box(ax, 2.5, 6.2, 5.0, 0.8, "Streamlit Web UI (EDA | CBF Search | Personal Rec)", C_NAVY, "white", 9)
    draw_box(ax, 1.5, 4.7, 2.5, 0.9, "preprocessing.py\n(Tiền xử lý)", C_BLUE, "white", 8)
    draw_box(ax, 4.5, 4.7, 2.5, 0.9, "evaluate.py\n(Độ đo đánh giá)", C_BLUE, "white", 8)
    draw_box(ax, 7.5, 4.7, 2.0, 0.9, "train_light.py\n(Huấn luyện)", C_BLUE, "white", 8)

    models = ["CBF", "User-CF", "Item-CF", "SVD", "NCF"]
    for i, m in enumerate(models):
        draw_box(ax, 1.0 + i * 1.7, 2.8, 1.5, 0.9, m, C_PURPLE, "white", 8)

    draw_box(ax, 1.5, 0.8, 3.0, 0.9, "foods.csv (4.000 món)", C_ORANGE, "white", 9)
    draw_box(ax, 5.0, 0.8, 3.5, 0.9, "ratings.csv (182.678 ratings)", C_ORANGE, "white", 9)

    for y_from, y_to in [(6.2, 5.15), (3.0, 1.7)]:
        ax.annotate("", xy=(5, y_to), xytext=(5, y_from),
                    arrowprops=dict(arrowstyle="<->", color=C_GRAY, lw=1.2, linestyle="dashed"))
    _save(fig, "system_architecture.png")


def fig_model_selection():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Quy trình Lựa chọn Mô hình Tối ưu (Model Selection)",
                 fontsize=13, fontweight="bold", color=C_NAVY, pad=12)

    draw_box(ax, 0.5, 3.5, 2.0, 1.0, "Huấn luyện\n4 phương pháp", C_ORANGE, "white", 9)
    draw_box(ax, 3.0, 3.5, 2.0, 1.0, "Đánh giá trên\nTest Set (10%)", C_BLUE, "white", 9)
    draw_box(ax, 5.5, 4.0, 1.8, 0.8, "Regression\n(RMSE, MAE)", C_PURPLE, "white", 8)
    draw_box(ax, 5.5, 3.0, 1.8, 0.8, "Ranking\n(P@10, NDCG)", C_PURPLE, "white", 8)
    draw_box(ax, 7.8, 3.5, 1.8, 1.0, "So sánh\n& Chọn Best", C_GREEN, "white", 9)

    draw_box(ax, 1.0, 1.0, 3.5, 1.2,
             "Rating Prediction:\n→ Neural CF (RMSE=1.59, MAE=1.38)", C_NAVY, "white", 9)
    draw_box(ax, 5.0, 1.0, 4.0, 1.2,
             "Top-K Ranking:\n→ User-CF Cosine (P@10=39.5%, NDCG=39.9%)", C_NAVY, "white", 9)

    draw_arrow(ax, 2.5, 4.0, 3.0, 4.0)
    draw_arrow(ax, 5.0, 4.0, 5.5, 4.4)
    draw_arrow(ax, 5.0, 4.0, 5.5, 3.4)
    draw_arrow(ax, 7.3, 4.0, 7.8, 4.0)
    draw_arrow(ax, 8.7, 3.5, 2.75, 2.2)
    draw_arrow(ax, 8.7, 3.5, 7.0, 2.2)
    _save(fig, "model_selection_process.png")


def fig_eda_from_data():
    foods_path = os.path.join(PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "Clean Dataset", "foods.csv")
    ratings_path = os.path.join(PROJECT_ROOT, "Clean Dataset-20260614T040023Z-3-001", "Clean Dataset", "ratings.csv")
    df_f = pd.read_csv(foods_path)
    df_r = pd.read_csv(ratings_path)

    # Dish type distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    types = df_f["dish_type"].value_counts().head(8)
    colors = sns.color_palette("mako", len(types))
    bars = ax.barh(types.index[::-1], types.values[::-1], color=colors[::-1])
    ax.set_xlabel("Số lượng món ăn")
    ax.set_title("Phân phối Loại Món ăn trong ViFoodRec", fontweight="bold", color=C_NAVY)
    for bar, val in zip(bars, types.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, "dish_type_distribution.png")

    # User rating activity
    fig, ax = plt.subplots(figsize=(8, 5))
    user_counts = df_r.groupby("userId").size()
    ax.hist(user_counts, bins=20, color=C_BLUE, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Số lượng ratings mỗi người dùng")
    ax.set_ylabel("Số người dùng")
    ax.set_title("Phân phối Mức độ Hoạt động Người dùng", fontweight="bold", color=C_NAVY)
    ax.axvline(user_counts.mean(), color=C_ORANGE, linestyle="--", label=f"Trung bình: {user_counts.mean():.0f}")
    ax.legend()
    plt.tight_layout()
    _save(fig, "user_activity_distribution.png")

    # Rating matrix heatmap (sample)
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_users = sorted(df_r["userId"].unique())[:15]
    sample_foods = sorted(df_r["foodId"].unique())[:20]
    matrix = np.zeros((len(sample_users), len(sample_foods)))
    sub = df_r[df_r["userId"].isin(sample_users) & df_r["foodId"].isin(sample_foods)]
    u_map = {u: i for i, u in enumerate(sample_users)}
    f_map = {f: j for j, f in enumerate(sample_foods)}
    for _, row in sub.iterrows():
        matrix[u_map[row["userId"]], f_map[row["foodId"]]] = row["rating"]
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=5)
    ax.set_xlabel("Food ID (sample)")
    ax.set_ylabel("User ID (sample)")
    ax.set_title("Ma trận Rating (Mẫu 15 Users × 20 Foods)", fontweight="bold", color=C_NAVY)
    plt.colorbar(im, ax=ax, label="Rating")
    plt.tight_layout()
    _save(fig, "rating_matrix_sample.png")

    # Nutrition scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_f["calories"], df_f["protein"], alpha=0.4, c=df_f["fat"], cmap="viridis", s=15)
    ax.set_xlabel("Calories (kcal)")
    ax.set_ylabel("Protein (g)")
    ax.set_title("Phân tích Dinh dưỡng Món ăn (Calories vs Protein)", fontweight="bold", color=C_NAVY)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Fat (g)")
    plt.tight_layout()
    _save(fig, "nutrition_analysis.png")

    # Stats summary table as figure
    stats = {
        "Chỉ số": ["Số món ăn", "Số người dùng", "Số ratings", "Density", "Rating TB",
                   "Calories TB", "Thời gian nấu TB"],
        "Giá trị": [
            f"{len(df_f):,}", f"{df_r['userId'].nunique():,}", f"{len(df_r):,}",
            f"{len(df_r) / (df_r['userId'].nunique() * len(df_f)) * 100:.2f}%",
            f"{df_r['rating'].mean():.2f}", f"{df_f['calories'].mean():.0f} kcal",
            f"{df_f['cooking_time'].mean():.0f} phút",
        ],
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    table = ax.table(cellText=list(zip(stats["Chỉ số"], stats["Giá trị"])),
                     colLabels=["Chỉ số", "Giá trị"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor(C_NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(C_LIGHT)
    ax.set_title("Thống kê Mô tả Bộ dữ liệu ViFoodRec", fontweight="bold", color=C_NAVY, pad=20)
    _save(fig, "dataset_statistics.png")


def fig_per_method_comparison():
    reg = pd.read_csv(os.path.join(FIGURES_DIR, "regression_metrics.csv"), index_col=0)
    rank = pd.read_csv(os.path.join(FIGURES_DIR, "ranking_metrics.csv"), index_col=0)

    groups = {
        "Content-Based": ["Content-Based"],
        "Collaborative Filtering": ["User-CF (Cosine)", "User-CF (Pearson)", "Item-CF (Cosine)", "Item-CF (Pearson)"],
        "Funk SVD": ["Funk SVD (MF)"],
        "Neural CF": ["Neural CF (NCF)"],
    }
    colors_map = {"Content-Based": C_ORANGE, "Collaborative Filtering": C_BLUE,
                  "Funk SVD": C_PURPLE, "Neural CF": C_GREEN}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Kết quả Đánh giá theo từng Nhóm Phương pháp", fontsize=14, fontweight="bold", color=C_NAVY)

    for ax, (group_name, models) in zip(axes.flat, groups.items()):
        sub_reg = reg.loc[models]
        x = np.arange(len(sub_reg))
        w = 0.35
        ax.bar(x - w / 2, sub_reg["RMSE"], w, label="RMSE", color=colors_map[group_name], alpha=0.8)
        ax.bar(x + w / 2, sub_reg["MAE"], w, label="MAE", color=C_ORANGE, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=7, rotation=0)
        ax.set_title(group_name, fontweight="bold", fontsize=11)
        ax.set_ylabel("Error ↓")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, "per_method_evaluation.png")


def main():
    print("=== Generating academic figures ===")
    fig_data_collection()
    fig_textual_eda()
    fig_ratings_before_after()
    fig_preprocessing_pipeline()
    fig_research_framework()
    fig_method_architecture("Content-Based Filtering (CBF)",
                            "cbf_architecture.png",
                            [("Input", "Metadata\nmón ăn"), ("TF-IDF", "Vector\nhóa"), ("User\nProfile", "Trung bình\ncó trọng số"), ("Cosine", "Similarity"), ("Output", "Rating\n/ Top-K")])
    fig_method_architecture("Collaborative Filtering (CF)",
                            "cf_architecture.png",
                            [("Rating\nMatrix", "101×4000"), ("Similarity", "Cosine/\nPearson"), ("Top-K\nNeighbors", "k=15"), ("Predict", "Weighted\nAvg"), ("Output", "Rating")])
    fig_method_architecture("Funk SVD (Matrix Factorization)",
                            "svd_architecture.png",
                            [("Input", "User-Item\nPair"), ("Embedding", "P_u, Q_i"), ("Bias", "b_u, b_i"), ("SGD", "Optimize"), ("Output", "Rating")])
    fig_method_architecture("Neural CF (NCF)",
                            "ncf_architecture.png",
                            [("Input", "User ID\nFood ID"), ("Embed", "16-dim"), ("MLP", "[32,16]"), ("ReLU+\nDropout", "Train"), ("Output", "Rating")])
    fig_cbf_recommendation_process()
    fig_food_recommendation_system()
    fig_recommendation_process()
    fig_system_architecture()
    fig_model_selection()
    fig_eda_from_data()
    if os.path.exists(os.path.join(FIGURES_DIR, "regression_metrics.csv")):
        fig_per_method_comparison()
    print("All figures generated!")


if __name__ == "__main__":
    main()
