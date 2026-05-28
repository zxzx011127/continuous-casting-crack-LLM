import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from openai import OpenAI
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 1. ⚙️ 配置区 =================
API_KEY = "sk-6b72c671e186497390da54bed9b38593" # 替换为真实 API Key
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 🌟 当前要评测的模型结果文件 (确保文件里已有 instruction, input, ground_truth, model_prediction)
INPUT_JSON = 'result_OpenBook_DeepSeek-R1-Distill-Qwen-7B_with_input.json' 

# 输出文件命名前缀
MODEL_LABEL = "DeepSeek-R1-Distill-Qwen-7B_with_input_OpenBook"
OUTPUT_EXCEL = f'Evaluation_{MODEL_LABEL}.xlsx'
OUTPUT_RADAR_DETAIL = f'Radar_Detail_{MODEL_LABEL}.png'
OUTPUT_RADAR_SUMMARY = f'Radar_Summary_{MODEL_LABEL}.png'

SKIP_INDICES = []
MAX_WORKERS = 3 

WEIGHTS = {
    "Dim1_Process": 0.35,
    "Dim2_Material": 0.35,
    "Dim3_AI_Logic": 0.30
}

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 2. 📊 评价体系 =================
DIMENSIONS = {
    "Dim1_Process": {
        "name": "Process & Operational Feasibility",
        "experts": ["Casting_Process_Expert", "Thermo_Mechanics_Analyst"],
        "angles": {
            "Defect_Causality": (
                "Defect Causality: Evaluate the logical chain from the given conditions to macroscopic crack initiation. "
                "Strict Rule: Evaluate ONLY based on the scope of the prompt. Do not penalize the model for omitting upstream "
                "process anomalies if the prompt only asks about downstream or micro-mechanisms."
            ),
            "Stress_Strain_Logic": (
                "Stress Analysis: Accuracy of explaining relevant thermo-mechanical forces (e.g., thermal contraction, "
                "bending stress, or mold friction). Strict Rule: Do NOT expect all force types to be mentioned. "
                "Only assess the accuracy of the specific forces relevant to the specific crack scenario described."
            ),
            "Operational_Feasibility": (
                "Operational Feasibility (Conditional): IF the model proactively suggests process optimizations, evaluate their "
                "practicality and safety for a real steel plant. IF the model purely provides a mechanistic analysis (because "
                "the prompt did not explicitly ask for a solution), DO NOT penalize it. Instead, score high if the mechanistic "
                "analysis is clear and diagnostic enough to implicitly guide on-site engineers."
            )
        }
    },
    "Dim2_Material": {
        "name": "Material Science & Microstructure",
        "experts": ["Physical_Metallurgist", "Microstructure_Specialist"],
        "angles": {
            "Solidification_Dynamics": (
                "Phase Dynamics & Thermal History: Accuracy of discussing relevant metallurgical transformations. "
                "Strict Rule: This covers BOTH liquid-to-solid solidification AND solid-state phase transformations "
                "(e.g., austenite to ferrite). Do not penalize the omission of 'solidification' if the defect occurs "
                "purely in the solid state (e.g., during straightening)."
            ),
            "Micro_Heterogeneity": (
                "Micro-Heterogeneity: Accurate analysis of localized weak points such as element segregation, "
                "precipitates (e.g., carbonitrides), or grain boundary coarsening, relevant to the specific steel grade."
            ),
            "Crack_Morphology": (
                "Crack Morphology: Correct physical correlation between crack type (e.g., intergranular vs. transgranular) "
                "and its propagation path along microstructural weaknesses."
            )
        }
    },
    "Dim3_AI_Logic": {
        "name": "AI Quality & Fidelity",
        "experts": ["AI_Fidelity_Auditor", "Academic_Linguist"],
        "angles": {
            "Fact_Fidelity": (
                "Factual Fidelity: Scientific accuracy according to fundamental metallurgical thermodynamics and physics. "
                "Check for 'Numerical Hallucinations' (e.g., completely wrong phase transformation temperatures)."
            ),
            "Reasoning_Depth": (
                "Reasoning Depth: Ability to perform multi-hop logical deduction (A causes B, B causes C). "
                "For context-provided (Open-Book) scenarios, reward synthesis over superficial copying. "
                "For no-context (Closed-Book) scenarios, reward independent and coherent derivation from first principles."
            ),
            "Term_Precision": (
                "Terminology: Precise, professional use of English metallurgical terminology. "
                "Strict Rule: Do not penalize minor phrasing differences if the underlying physical meaning is completely accurate."
            )
        }
    }
}

# ================= 3. 🧠 Prompt 构建器 (开卷版) =================
def build_prompt(dim_key, expert_role):
    dim_info = DIMENSIONS[dim_key]
    criteria_text = "\n".join([f"- {k}: {v}" for k, v in dim_info['angles'].items()])
    
    angle_keys = list(dim_info['angles'].keys())
    json_structure = "{\n"
    for k in angle_keys:
        json_structure += f'  "{k}_Score": float (1.0 to 5.0),\n'
        json_structure += f'  "{k}_Reason": "brief justification in English",\n'
    json_structure += "}"

    prompt = f"""
    You are a STRICT and HIGHLY PROFESSIONAL {expert_role} reviewing an AI model's answer about Continuous Casting Surface Cracks.
    You will be provided with:
    1. The original Question.
    2. The Context (provided to the AI).
    3. The Ground Truth (Standard Reference Answer).
    4. The Model's Answer to be evaluated.

    Evaluate the Model's Answer based on these 3 criteria:
    {criteria_text}
    
    *** REFINED SCORING RULES (Expert & Rigorous Version) ***
    1. **Format Exemption (Crucial)**: The model may or may not output its reasoning inside <think>...</think> tags. DO NOT penalize the absence of these tags. Evaluate the logical reasoning and final answer strictly based on the text provided.
    2. **Contextual Faithfulness & Anti-Distraction**: Check if the model accurately extracted information from the Context. If the Context contains distractors (e.g., mentioning AlN instead of TiN), and the model reasonably addresses them based ONLY on the provided text, do NOT penalize it.
    3. **In-Depth Mechanism Reward**: If the model invokes metallurgical principles not explicitly in the Context, it MUST be evaluated for SCIENTIFIC ACCURACY. 
       - Correct extension = HIGH score (4.0-5.0).
       - Vague extension = MID score (3.0).
       - Scientifically WRONG extension = LOW score (1.0-2.0).
    4. **Term Precision**: Penalize vague descriptions. A high-quality answer should specify the microstructural weak points.
    
    Return ONLY a JSON object with these exact keys:
    {json_structure}
    """
    return prompt

def call_expert_api(system_prompt, item):
    question = item.get('instruction', 'N/A')
    context = item.get('input', 'N/A') # 👈 精准抓取 input 字段
    ground_truth = item.get('ground_truth', 'N/A')
    model_answer = item.get('model_prediction', 'N/A')

    user_content = (
        f"Question: {question}\n\n"
        f"Context: {context}\n\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"Model Answer: {model_answer}"
    )
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={'type': 'json_object'},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")
        return {}

def evaluate_task(idx, item):
    task_result = {"id": idx, "dim_results": {}, "summary_avgs": {}}
    for dim_key, dim_conf in DIMENSIONS.items():
        dim_data = {}
        dim_total = 0
        count = 0
        for expert in dim_conf['experts']:
            prompt = build_prompt(dim_key, expert)
            res = call_expert_api(prompt, item)
            for angle_key in dim_conf['angles'].keys():
                score = res.get(f"{angle_key}_Score", 0)
                dim_data[f"{expert}_{angle_key}"] = score
                dim_data[f"{expert}_{angle_key}_Comment"] = res.get(f"{angle_key}_Reason", "Parsing Failed")
                dim_total += score
                count += 1
        
        avg = round(dim_total / max(count, 1), 2)
        dim_data["Dimension_Average"] = avg
        task_result["dim_results"][dim_key] = dim_data
        task_result["summary_avgs"][dim_key] = avg
    return task_result

def add_summary_row(df):
    if df.empty: return df
    numeric_cols = df.select_dtypes(include=['number']).columns
    means = df[numeric_cols].mean().round(2)
    summary_row = {col: "" for col in df.columns}
    summary_row.update(means.to_dict())
    if "ID" in df.columns: summary_row["ID"] = "TOTAL_AVG"
    if "Question_Full" in df.columns: summary_row["Question_Full"] = "GLOBAL AVERAGE"
    return pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

def plot_detailed_radar(final_data_dict, output_path):
    labels = [k.replace("_", "\n") for k in final_data_dict.keys()]
    values = list(final_data_dict.values())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#1f77b4', linewidth=2, marker='o', label=MODEL_LABEL)
    ax.fill(angles, values, color='#1f77b4', alpha=0.2)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 5)
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(angle, value + 0.3, f"{value:.2f}", ha='center', fontsize=10, weight='bold', color='#1f77b4')
    plt.title(f"Detailed Expert Evaluation - {MODEL_LABEL}", size=15, y=1.08, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

def plot_summary_radar(dim_scores_dict, output_path):
    labels = [d['name'].replace(" & ", "\n&\n") for d in DIMENSIONS.values()]
    values = list(dim_scores_dict.values())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='#ff7f0e', linewidth=3, marker='D', label='Overall')
    ax.fill(angles, values, color='#ff7f0e', alpha=0.2)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, weight='bold')
    ax.set_ylim(0, 5)
    for angle, value in zip(angles[:-1], values[:-1]):
        ax.text(angle, value + 0.4, f"{value:.2f}", ha='center', color='#ff7f0e', weight='bold', fontsize=11)
    plt.title(f"Dimension Summary - {MODEL_LABEL}", size=14, y=1.1, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

# ================= 4. 🚀 主执行程序 =================
def main():
    if not os.path.exists(INPUT_JSON):
        print(f"❌ 错误: 找不到输入文件 {INPUT_JSON}")
        return

    # 🌟 直接读取缝合好的 JSON
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = [item for i, item in enumerate(data) if i not in SKIP_INDICES]
    storage = {"Dim1_Process": [], "Dim2_Material": [], "Dim3_AI_Logic": [], "Final_Summary": []}
    global_angle_scores = {k: [] for d in DIMENSIONS.values() for k in d['angles'].keys()}
    global_dim_scores = {k: [] for k in DIMENSIONS.keys()}

    print(f"\n🚀 开始执行开卷并发评测 | 并发数: {MAX_WORKERS} | 当前模型: {MODEL_LABEL}\n")

    all_task_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(evaluate_task, i, item): i for i, item in enumerate(filtered_data)}
        for future in as_completed(futures):
            res = future.result()
            all_task_results.append(res)
            print(f"✅ 已完成第 {res['id']+1} 题的专家评审")

    all_task_results.sort(key=lambda x: x["id"])

    for i, res in enumerate(all_task_results):
        item = filtered_data[i]
        q_text = item.get('instruction', 'N/A')
        a_text = item.get('model_prediction', 'N/A')
        base_info = {"ID": res["id"]+1, "Question_Full": q_text, "Model_Answer_Full": a_text}

        weighted_score = 0
        for dim_key, weight in WEIGHTS.items():
            row_data = {**base_info, **res["dim_results"][dim_key]}
            storage[dim_key].append(row_data)
            global_dim_scores[dim_key].append(res["summary_avgs"][dim_key])
            for k in DIMENSIONS[dim_key]['angles'].keys():
                for expert in DIMENSIONS[dim_key]['experts']:
                    score = res["dim_results"][dim_key].get(f"{expert}_{k}", 0)
                    if score > 0: global_angle_scores[k].append(score)
            weighted_score += res["summary_avgs"][dim_key] * weight

        summary_row = {**base_info, "Final_Weighted_Score": round(weighted_score, 2)}
        for d_key in DIMENSIONS: summary_row[f"{d_key}_Avg"] = res["summary_avgs"][d_key]
        storage["Final_Summary"].append(summary_row)

    print("\n✅ 评测完成！正在生成数据报表与雷达图...")
    plot_detailed_radar({k: np.mean(v) if v else 0 for k, v in global_angle_scores.items()}, OUTPUT_RADAR_DETAIL)
    plot_summary_radar({k: np.mean(v) if v else 0 for k, v in global_dim_scores.items()}, OUTPUT_RADAR_SUMMARY)

    df_p = add_summary_row(pd.DataFrame(storage["Dim1_Process"]))
    df_m = add_summary_row(pd.DataFrame(storage["Dim2_Material"]))
    df_a = add_summary_row(pd.DataFrame(storage["Dim3_AI_Logic"]))
    df_s = add_summary_row(pd.DataFrame(storage["Final_Summary"]))

    rules = []
    for d_val in DIMENSIONS.values():
        for k, v in d_val['angles'].items():
            rules.append({"Dimension": d_val['name'], "Sub_Angle": k, "Description": v})
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df_s.to_excel(writer, sheet_name='Summary_Scores', index=False)
        df_m.to_excel(writer, sheet_name='Material_Science', index=False)
        df_p.to_excel(writer, sheet_name='Process_Mechanics', index=False)
        df_a.to_excel(writer, sheet_name='AI_Logic', index=False)
        pd.DataFrame(rules).to_excel(writer, sheet_name='Evaluation_Rules', index=False)

    print(f"🎉 恭喜！开卷结果已成功导出至 Excel: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()