import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def update_latex():
    # Define file paths
    reg_csv_path = os.path.join(PROJECT_ROOT, "report", "figures", "regression_metrics.csv")
    rank_csv_path = os.path.join(PROJECT_ROOT, "report", "figures", "ranking_metrics.csv")
    tex_path = os.path.join(PROJECT_ROOT, "report", "report.tex")
    
    if not os.path.exists(reg_csv_path) or not os.path.exists(rank_csv_path):
        print("Error: Metrics CSV files not found! Make sure you run train.py first.")
        return
        
    # Load CSVs
    df_reg = pd.read_csv(reg_csv_path).set_index('Unnamed: 0')
    df_rank = pd.read_csv(rank_csv_path).set_index('Unnamed: 0')
    
    # Load latex file
    with open(tex_path, 'r', encoding='utf-8') as f:
        tex_content = f.read()
        
    # Define mapping of CSV model names to LaTeX placeholder prefixes
    name_map = {
        "Content-Based": "CBF",
        "User-CF (Cosine)": "UCF_COS",
        "User-CF (Pearson)": "UCF_PEA",
        "Item-CF (Cosine)": "ICF_COS",
        "Item-CF (Pearson)": "ICF_PEA",
        "Funk SVD (MF)": "SVD",
        "Neural CF (NCF)": "NCF"
    }
    
    # Replace regression placeholders
    for model_name, prefix in name_map.items():
        if model_name in df_reg.index:
            row = df_reg.loc[model_name]
            tex_content = tex_content.replace(f"[{prefix}_MSE]", f"{row['MSE']:.4f}")
            tex_content = tex_content.replace(f"[{prefix}_RMSE]", f"{row['RMSE']:.4f}")
            tex_content = tex_content.replace(f"[{prefix}_MAE]", f"{row['MAE']:.4f}")
            tex_content = tex_content.replace(f"[{prefix}_NMAE]", f"{row['NMAE']:.4f}")
            
    # Replace ranking placeholders
    for model_name, prefix in name_map.items():
        if model_name in df_rank.index:
            row = df_rank.loc[model_name]
            tex_content = tex_content.replace(f"[{prefix}_PREC]", f"{row['Precision@K']:.2%}")
            tex_content = tex_content.replace(f"[{prefix}_REC]", f"{row['Recall@K']:.2%}")
            tex_content = tex_content.replace(f"[{prefix}_NDCG]", f"{row['NDCG@K']:.2%}")
            tex_content = tex_content.replace(f"[{prefix}_MRR]", f"{row['MRR']:.2%}")
            
    # Save latex file
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    print("LaTeX file updated successfully with training metrics.")

if __name__ == "__main__":
    update_latex()
