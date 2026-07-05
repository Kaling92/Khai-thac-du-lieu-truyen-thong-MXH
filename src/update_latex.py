import os
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def format_value(val, is_pct=False, is_bold=False):
    if is_pct:
        formatted = f"{val * 100:.2f}\\%"
    else:
        formatted = f"{val:.4f}"
    
    if is_bold:
        return f"\\textbf{{{formatted}}}"
    return formatted

def generate_row_string(model_label, model_name, df_reg, df_rank, best_reg, best_rank):
    reg_row = df_reg.loc[model_name]
    rank_row = df_rank.loc[model_name]
    
    mse = format_value(reg_row['MSE'], False, best_reg['MSE'] == model_name)
    rmse = format_value(reg_row['RMSE'], False, best_reg['RMSE'] == model_name)
    mae = format_value(reg_row['MAE'], False, best_reg['MAE'] == model_name)
    nmae = format_value(reg_row['NMAE'], False, best_reg['NMAE'] == model_name)
    
    prec = format_value(rank_row['Precision@K'], True, best_rank['Precision@K'] == model_name)
    rec = format_value(rank_row['Recall@K'], True, best_rank['Recall@K'] == model_name)
    ndcg = format_value(rank_row['NDCG@K'], True, best_rank['NDCG@K'] == model_name)
    mrr = format_value(rank_row['MRR'], True, best_rank['MRR'] == model_name)
    
    return f"{model_label} & {mse} & {rmse} & {mae} & {nmae} & {prec} & {rec} & {ndcg} & {mrr} \\\\"

def update_latex():
    reg_csv_path = os.path.join(PROJECT_ROOT, "report", "figures", "regression_metrics.csv")
    rank_csv_path = os.path.join(PROJECT_ROOT, "report", "figures", "ranking_metrics.csv")
    tex_path = os.path.join(PROJECT_ROOT, "report", "report.tex")
    
    if not os.path.exists(reg_csv_path) or not os.path.exists(rank_csv_path):
        print("Error: Metrics CSV files not found! Make sure you run train.py first.")
        return
        
    df_reg = pd.read_csv(reg_csv_path, index_col=0)
    df_rank = pd.read_csv(rank_csv_path, index_col=0)
    
    # Define model groups
    all_models = list(df_reg.index)
    cf_models = ["User-CF (Cosine)", "User-CF (Pearson)", "Item-CF (Cosine)", "Item-CF (Pearson)"]
    overall_models = ["Content-Based", "User-CF (Cosine)", "Item-CF (Cosine)", "Funk SVD (MF)", "Neural CF (NCF)"]
    
    # Helper to find best models in a group
    def get_best_models(group):
        best_reg = {}
        for col in ['MSE', 'RMSE', 'MAE', 'NMAE']:
            best_reg[col] = df_reg.loc[group, col].idxmin()
            
        best_rank = {}
        for col in ['Precision@K', 'Recall@K', 'NDCG@K', 'MRR']:
            best_rank[col] = df_rank.loc[group, col].idxmax()
            
        return best_reg, best_rank

    best_all_reg, best_all_rank = get_best_models(all_models)
    best_cf_reg, best_cf_rank = get_best_models(cf_models)
    best_overall_reg, best_overall_rank = get_best_models(overall_models)

    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()
        
    # 1. Update Bảng 3: CBF
    cbf_best_reg = {'MSE': "Content-Based", 'RMSE': "Content-Based", 'MAE': "Content-Based", 'NMAE': "Content-Based"}
    cbf_best_rank = {'Precision@K': "Content-Based", 'Recall@K': "Content-Based", 'NDCG@K': "Content-Based", 'MRR': "Content-Based"}
    cbf_row = generate_row_string("CBF (TF-IDF)", "Content-Based", df_reg, df_rank, cbf_best_reg, cbf_best_rank)
    tex_content = re.sub(r'CBF \(TF-IDF\)\s*&.*?\\\\', lambda m: cbf_row, tex_content)
    
    # 2. Update Bảng 4: CF (local bolding)
    for model_name, latex_label in [
        ("User-CF (Cosine)", "User-CF (Cosine)"),
        ("User-CF (Pearson)", "User-CF (Pearson)"),
        ("Item-CF (Cosine)", "Item-CF (Cosine)"),
        ("Item-CF (Pearson)", "Item-CF (Pearson)")
    ]:
        row_str = generate_row_string(latex_label, model_name, df_reg, df_rank, best_cf_reg, best_cf_rank)
        # Match only inside the Collaborative Filtering table, non-greedy
        pattern = re.escape(latex_label) + r'\s*&.*?\\\\'
        tex_content = re.sub(pattern, lambda m, rs=row_str: rs, tex_content)

    # 3. Update Bảng 5: Funk SVD
    svd_best_reg = {'MSE': "Funk SVD (MF)", 'RMSE': "Funk SVD (MF)", 'MAE': "Funk SVD (MF)", 'NMAE': "Funk SVD (MF)"}
    svd_best_rank = {'Precision@K': "Funk SVD (MF)", 'Recall@K': "Funk SVD (MF)", 'NDCG@K': "Funk SVD (MF)", 'MRR': "Funk SVD (MF)"}
    svd_row = generate_row_string("Funk SVD ($f$=10)", "Funk SVD (MF)", df_reg, df_rank, svd_best_reg, svd_best_rank)
    tex_content = re.sub(r'Funk SVD \(\$f\$=10\)\s*&.*?\\\\', lambda m: svd_row, tex_content)

    # 4. Update Bảng 6: NCF
    ncf_best_reg = {'MSE': "Neural CF (NCF)", 'RMSE': "Neural CF (NCF)", 'MAE': "Neural CF (NCF)", 'NMAE': "Neural CF (NCF)"}
    ncf_best_rank = {'Precision@K': "Neural CF (NCF)", 'Recall@K': "Neural CF (NCF)", 'NDCG@K': "Neural CF (NCF)", 'MRR': "Neural CF (NCF)"}
    ncf_row = generate_row_string("NCF (MLP)", "Neural CF (NCF)", df_reg, df_rank, ncf_best_reg, ncf_best_rank)
    tex_content = re.sub(r'NCF \(MLP\)\s*&.*?\\\\', lambda m: ncf_row, tex_content)

    # 5. Update Bảng 7: Overall (bold best overall among overall_models)
    overall_block_match = re.search(r'(\\label{tab:overall}.*?\\begin{tabular}{lcccc\|cccc}.*?\\midrule)(.*?)(\\bottomrule)', tex_content, re.DOTALL)
    if overall_block_match:
        header_part = overall_block_match.group(1)
        body_part = overall_block_match.group(2)
        footer_part = overall_block_match.group(3)
        
        new_overall_rows = []
        
        # Row 1: Content-Based
        row1 = generate_row_string("Content-Based", "Content-Based", df_reg, df_rank, best_overall_reg, best_overall_rank)
        new_overall_rows.append(row1)
        
        # Row 2: User-CF (Cosine)
        row2 = generate_row_string("User-CF (Cosine)", "User-CF (Cosine)", df_reg, df_rank, best_overall_reg, best_overall_rank)
        new_overall_rows.append(row2)
        
        # Row 3: Item-CF (Cosine)
        row3 = generate_row_string("Item-CF (Cosine)", "Item-CF (Cosine)", df_reg, df_rank, best_overall_reg, best_overall_rank)
        new_overall_rows.append(row3)
        
        # Row 4: Funk SVD
        row4 = generate_row_string("Funk SVD", "Funk SVD (MF)", df_reg, df_rank, best_overall_reg, best_overall_rank)
        new_overall_rows.append(row4)
        
        # Row 5: Neural CF
        ncf_label = "\\textbf{Neural CF}" if best_overall_reg['RMSE'] == "Neural CF (NCF)" else "Neural CF"
        row5 = generate_row_string(ncf_label, "Neural CF (NCF)", df_reg, df_rank, best_overall_reg, best_overall_rank)
        new_overall_rows.append(row5)
        
        new_body = "\n" + "\n".join(new_overall_rows) + "\n"
        
        # Replace the body in tex_content
        full_old_block = overall_block_match.group(0)
        full_new_block = header_part + new_body + footer_part
        tex_content = tex_content.replace(full_old_block, full_new_block)
        
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    print("LaTeX report tables successfully updated with deterministic seeded results!")

if __name__ == "__main__":
    update_latex()
