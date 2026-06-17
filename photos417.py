import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

# ==========================================
# 1. 映射配置：文件名关键词 -> 论文显示的正式名称
# ==========================================
MODEL_MAP = {
    'sft_3_4': 'SFT 3-4',
    'DeepSeek-R1-Distill-Qwen-7B': 'DeepSeek-R1-Distill-Qwen-7B',
    'Llama3.1-8B-Instruct': 'Llama3.1-8B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B-Instruct'
}

# 定义三大维度（主维度）
SUMMARY_DIMENSIONS = {
    'Dim1_Process_Avg': 'Dim 1: Process',
    'Dim2_Material_Avg': 'Dim 2: Material',
    'Dim3_AI_Logic_Avg': 'Dim 3: AI Logic'
}

# 定义九大子指标（细节维度）
DETAIL_METRICS = {
    'Process_Mechanics': ['Defect_Causality', 'Stress_Strain_Logic', 'Operational_Feasibility'],
    'Material_Science': ['Solidification_Dynamics', 'Micro_Heterogeneity', 'Crack_Morphology'],
    'Parametric_Logic': ['Fact_Fidelity', 'Reasoning_Depth', 'Term_Precision']
}

# ==========================================
# 2. 数据抓取核心函数
# ==========================================
def get_total_avg_score(df, keyword):
    row = df[df['ID'] == 'TOTAL_AVG']
    if row.empty: return 0
    cols = [c for c in df.columns if keyword in c and '_Comment' not in c]
    if not cols: return 0
    return pd.to_numeric(row[cols].iloc[0], errors='coerce').mean()

def load_all_data():
    results = {}
    for file_key, display_name in MODEL_MAP.items():
        for mode in ['OpenBook', 'ClosedBook']:
            suffix = "_with_input_OpenBook.xlsx" if mode == 'OpenBook' else "_ClosedBook_Ablation.xlsx"
            filename = f"Evaluation_{file_key}{suffix}"
            
            if 'DeepSeek' in file_key and mode == 'ClosedBook':
                filename = f"Evaluation__DeepSeek-R1-Distill-Qwen-7B_ClosedBook_Ablation.xlsx"

            if not os.path.exists(filename):
                print(f"⚠️ 跳过：未找到文件 {filename}")
                continue

            print(f"📖 正在解析：{filename}")
            m_key = f"{display_name}_{mode}"
            results[m_key] = {}

            try:
                with pd.ExcelFile(filename) as xls:
                    sheet_names = xls.sheet_names
                    if 'Summary_Scores' in sheet_names:
                        df_sum = pd.read_excel(xls, 'Summary_Scores')
                        row_avg = df_sum[df_sum['ID'] == 'TOTAL_AVG'].iloc[0]
                        for col, label in SUMMARY_DIMENSIONS.items():
                            results[m_key][label] = float(row_avg[col])

                    for sheet, keywords in DETAIL_METRICS.items():
                        actual_sheet = sheet
                        if sheet == 'Parametric_Logic' and 'Parametric_Logic' not in sheet_names and 'AI_Logic' in sheet_names:
                            actual_sheet = 'AI_Logic'
                            
                        if actual_sheet in sheet_names:
                            df_detail = pd.read_excel(xls, actual_sheet)
                            for kw in keywords:
                                results[m_key][kw] = get_total_avg_score(df_detail, kw)
                        else:
                            print(f"  ❌ 找不到表单: {actual_sheet} (在文件 {filename} 中)")
            except Exception as e:
                print(f"❌ 解析 {filename} 出错: {e}")
    return results

# ==========================================
# 3. 绘图核心函数 (已优化颜色和线条粗细)
# ==========================================
def draw_radar(data_dict, labels, title, save_path, colors):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], [l.replace('_', ' ') for l in labels], size=10, weight='bold')
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=9)
    plt.ylim(0, 5)
    
    for i, (model_name, scores) in enumerate(data_dict.items()):
        values = scores + scores[:1]
        # 💡 修改点：线条加粗至 3.0，颜色填充透明度调为 0.15，确保线条极度清晰
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=3.0, label=model_name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)
    plt.title(title, size=15, pad=30, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 4. 运行生成
# ==========================================
if __name__ == "__main__":
    all_data = load_all_data()
    if not all_data:
        print("🛑 未能读取到任何数据，请检查文件名是否正确且脚本与xlsx在同一目录。")
        exit()

    os.makedirs("Radar_Charts_HighContrast", exist_ok=True)
    full_model_names = list(MODEL_MAP.values())
    detail_labels = [kw for sublist in DETAIL_METRICS.values() for kw in sublist]
    summary_labels = list(SUMMARY_DIMENSIONS.values())

    # 💡 修改点：采用高对比度的学术界标准色板 (深红, 藏蓝, 亮橙, 翠绿)
    palette = ['#D62728', '#1F77B4', '#FF7F0E', '#2CA02C'] 

    # 图1：所有模型开卷对比
    ob_sum = {m: [all_data[f"{m}_OpenBook"][l] for l in summary_labels] for m in full_model_names if f"{m}_OpenBook" in all_data}
    draw_radar(ob_sum, summary_labels, "Open-Book: Summary Scores Contrast", "Radar_Charts_HighContrast/1_All_OpenBook_Summary.png", palette)
    
    ob_det = {m: [all_data[f"{m}_OpenBook"][l] for l in detail_labels] for m in full_model_names if f"{m}_OpenBook" in all_data}
    draw_radar(ob_det, detail_labels, "Open-Book: Detailed Metrics Contrast", "Radar_Charts_HighContrast/2_All_OpenBook_Detailed.png", palette)

    # 图2：所有模型闭卷对比
    cb_sum = {m: [all_data[f"{m}_ClosedBook"][l] for l in summary_labels] for m in full_model_names if f"{m}_ClosedBook" in all_data}
    draw_radar(cb_sum, summary_labels, "Closed-Book: Summary Scores Contrast", "Radar_Charts_HighContrast/3_All_ClosedBook_Summary.png", palette)
    
    cb_det = {m: [all_data[f"{m}_ClosedBook"][l] for l in detail_labels] for m in full_model_names if f"{m}_ClosedBook" in all_data}
    draw_radar(cb_det, detail_labels, "Closed-Book: Detailed Metrics Contrast", "Radar_Charts_HighContrast/4_All_ClosedBook_Detailed.png", palette)

    # 图3：各模型开闭卷消融对比
    compare_colors = ['#1F77B4', '#FF7F0E'] # 高对比度的深蓝(开卷) vs 亮橙(闭卷)
    for m in full_model_names:
        if f"{m}_OpenBook" in all_data and f"{m}_ClosedBook" in all_data:
            m_comp = {
                f"{m} (Open)": [all_data[f"{m}_OpenBook"][l] for l in detail_labels],
                f"{m} (Closed)": [all_data[f"{m}_ClosedBook"][l] for l in detail_labels]
            }
            draw_radar(m_comp, detail_labels, f"Ablation: {m} (Open vs Closed)", f"Radar_Charts_HighContrast/Ablation_{m}.png", compare_colors)

    print("\n🎉 所有高对比度雷达图已生成至 'Radar_Charts_HighContrast' 文件夹！")
