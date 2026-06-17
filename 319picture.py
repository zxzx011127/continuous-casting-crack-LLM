import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 全局学术风格设置
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# 模型映射（直接使用当前目录下的 xlsx 文件名）
models = {
    "DeepSeek-R1-Distill-Qwen-7B": "Evaluation_DeepSeek-R1-Distill-Qwen-7B_with_input_OpenBook.xlsx",
    "Llama3.1-8B-Instruct": "Evaluation_Llama3.1-8B-Instruct_with_input_OpenBook.xlsx",
    "Qwen2.5-7B-Instruct": "Evaluation_Qwen2.5-7B-Instruct_with_input_OpenBook.xlsx",
    "SFT 3-4": "Evaluation_sft_3_4_with_input_OpenBook.xlsx"
}

# ==========================================
# 图1: 专家评分一致性散点图 (Inter-Rater Agreement)
# ==========================================
try:
    df_process = pd.read_excel("Evaluation_sft_3_4_ClosedBook_Ablation.xlsx", sheet_name="Process_Mechanics")
    # 提取过滤掉 TOTAL_AVG 行的数据
    df_process = df_process[df_process['ID'] != 'TOTAL_AVG']
    
    expert_a = pd.to_numeric(df_process['Casting_Process_Expert_Defect_Causality'], errors='coerce').dropna()
    expert_b = pd.to_numeric(df_process['Thermo_Mechanics_Analyst_Defect_Causality'], errors='coerce').dropna()

    plt.figure(figsize=(6, 6))
    # 加上些许抖动(jitter)防止点完全重合
    sns.regplot(x=expert_a + np.random.normal(0, 0.05, len(expert_a)), 
                y=expert_b + np.random.normal(0, 0.05, len(expert_b)), 
                scatter_kws={'alpha':0.6, 'color':'#d9534f'}, line_kws={'color':'#333333', 'linestyle':'--'})
    plt.plot([1, 5], [1, 5], ls=":", c=".3") # 对角线
    plt.title("Inter-Rater Agreement (Defect Causality)", pad=15, fontweight='bold')
    plt.xlabel("Casting Process Expert Score")
    plt.ylabel("Thermo-Mechanics Analyst Score")
    plt.xlim(1.5, 5.2)
    plt.ylim(1.5, 5.2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_Expert_Agreement.png", dpi=300)
    plt.close()
    print("✅ 图1 (专家一致性散点图) 生成成功！")
except Exception as e:
    print(f"❌ 图1 生成失败，报错信息: {e}")

# ==========================================
# 图2: 模型得分核密度估计图 (KDE Plot)
# ==========================================
try:
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 红色给SFT，其余基座冷色

    for (model_name, file_name), color in zip(models.items(), colors):
        if os.path.exists(file_name):
            df = pd.read_excel(file_name, sheet_name="Summary_Scores")
            df = df[df['ID'] != 'TOTAL_AVG']
            scores = pd.to_numeric(df["Final_Weighted_Score"], errors='coerce').dropna()
            sns.kdeplot(scores, label=model_name, fill=True, alpha=0.3, color=color, linewidth=2)
        else:
            print(f"⚠️ 找不到文件: {file_name}")

    plt.title("Probability Density Distribution of Final Weighted Scores", pad=15, fontweight='bold')
    plt.xlabel("Final Evaluation Score (Out of 5.0)")
    plt.ylabel("Density")
    plt.xlim(1.0, 5.5)
    plt.legend(title="Models", loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Fig_KDE_ScoreDistribution.png", dpi=300)
    plt.close()
    print("✅ 图2 (得分核密度估计图) 生成成功！")
except Exception as e:
    print(f"❌ 图2 生成失败，报错信息: {e}")

# ==========================================
# 图3: 核心能力子维度相关性热力图 (Correlation Heatmap)
# ==========================================
try:
    # 读取各Sheet，并过滤掉 TOTAL_AVG 行
    df_m = pd.read_excel("Evaluation_sft_3_4_ClosedBook_Ablation.xlsx", sheet_name="Material_Science")
    df_m = df_m[df_m['ID'] != 'TOTAL_AVG']
    
    df_p = pd.read_excel("Evaluation_sft_3_4_ClosedBook_Ablation.xlsx", sheet_name="Process_Mechanics")
    df_p = df_p[df_p['ID'] != 'TOTAL_AVG']
    
    df_a = pd.read_excel("Evaluation_sft_3_4_ClosedBook_Ablation.xlsx", sheet_name="Parametric_Logic")
    df_a = df_a[df_a['ID'] != 'TOTAL_AVG']

    def get_avg_series(df, keyword):
        cols = [c for c in df.columns if keyword in c and '_Comment' not in c]
        return pd.to_numeric(df[cols].mean(axis=1), errors='coerce')

    # 提取 9 个子维度双专家均值
    cor_data = pd.DataFrame({
        'Defect Causality': get_avg_series(df_p, 'Defect_Causality'),
        'Stress & Strain': get_avg_series(df_p, 'Stress_Strain_Logic'),
        'Oper. Feasibility': get_avg_series(df_p, 'Operational_Feasibility'),
        
        'Solidification Dyn.': get_avg_series(df_m, 'Solidification_Dynamics'),
        'Micro-Hetero.': get_avg_series(df_m, 'Micro_Heterogeneity'),
        'Crack Morph.': get_avg_series(df_m, 'Crack_Morphology'),
        
        'Fact Fidelity': get_avg_series(df_a, 'Fact_Fidelity'),
        'Reasoning Depth': get_avg_series(df_a, 'Reasoning_Depth'),
        'Term Precision': get_avg_series(df_a, 'Term_Precision')
    })

    corr_matrix = cor_data.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", 
                linewidths=.5, cbar_kws={"shrink": .8}, vmin=0, vmax=1)
    plt.title("Correlation Matrix of 9 Sub-dimensions (Finetuned Qwen)", pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("Fig_Correlation_Heatmap.png", dpi=300)
    plt.close()
    print("✅ 图3 (核心能力相关性热力图) 生成成功！")
except Exception as e:
    print(f"❌ 图3 生成失败，报错信息: {e}")
