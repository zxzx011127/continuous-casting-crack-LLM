import json
import os
import random
import glob
import time
from openai import OpenAI

# ==============================================================================
#  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  【配置区域】  ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
# ==============================================================================

# 1. API Key
API_KEY = "sk-60ceb5caab3840b797b7fe0a1bdc3275" 

# 2. DeepSeek 配置
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 3. 路径配置
INPUT_DIR = r"D:\课题组\中心偏析——表面裂纹markdown\entity_end2"
OUTPUT_DIR = r"D:\课题组\中心偏析——表面裂纹markdown\QA330"

# 4. 生成数量控制 (每篇论文)
MIN_QA_PER_PAPER = 5   
MAX_QA_PER_PAPER = 15  

# ==============================================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

class ContextBuilder:
    def __init__(self, paper_data):
        self.structure = paper_data['structure']
        self.total = len(self.structure)

    def get_window(self, center_idx, radius=2):
        """滑动窗口：保证语义连贯"""
        start = max(0, center_idx - radius)
        end = min(self.total, center_idx + radius + 1)
        context_str = ""
        for i in range(start, end):
            item = self.structure[i]
            if item.get('content_type') == 'invalid_chart': continue
            
            marker = "[Chart/Data Analysis]" if item.get('content_type') == 'chart_description' else ""
            context_str += f"[Section: {item['section_hierarchy']}] {marker}{item['text']}\n"
        return context_str

    def get_multihop_context(self, idx_a, idx_b, radius=1):
        """多跳集成：物理拼接两个逻辑点"""
        block_a = self.get_window(idx_a, radius)
        block_b = self.get_window(idx_b, radius)
        if abs(idx_a - idx_b) < (radius * 2 + 1):
            return self.get_window(idx_a, radius + abs(idx_a - idx_b))
        return f"--- Context Fragment A (Cause/Condition) ---\n{block_a}\n......\n--- Context Fragment B (Effect/Result) ---\n{block_b}"

class StrategySampler:
    def __init__(self, paper_data):
        self.paper_id = paper_data.get('paper_id', 'unknown')
        self.structure = paper_data['structure']
        self.ctx = ContextBuilder(paper_data)
        
        self.idx_by_category = {} 
        self.chart_indices = []
        self.conclusion_indices = []
        self.anomaly_indices = []
        
        # 【修改点 1】：关键词转向表面裂纹与结晶器/二冷区异常
        anomaly_keywords = ["breakout", "sticker", "bleeding", "alarm", "abnormal", "crack", "depression", "oscillation mark", "hot tear"]

        for item in self.structure:
            idx = item['index']
            text_lower = item['text'].lower()
            ctype = item.get('content_type')
            
            if ctype in ['chart_description', 'formula_description']:
                if ctype != 'invalid_chart' and len(item.get('entities', [])) > 0:
                    self.chart_indices.append(idx)
            
            if 'conclusion' in item['section_hierarchy'].lower():
                self.conclusion_indices.append(idx)
            
            if any(k in text_lower for k in anomaly_keywords):
                self.anomaly_indices.append(idx)

            for ent in item.get('entities', []):
                cat = ent['category']
                if cat not in self.idx_by_category: self.idx_by_category[cat] = []
                self.idx_by_category[cat].append(idx)

    def get_tasks(self):
        """生成任务候选池 (完全保留你的原逻辑)"""
        tasks = []
        
        # 1. [变量影响结果] 
        params = self.idx_by_category.get('PARAMETER', [])
        defects = self.idx_by_category.get('DEFECT', [])
        if params and defects:
            pairs = []
            for _ in range(30):
                p = random.choice(params)
                d = random.choice(defects)
                # 仅将这里修改为绝对值，防止倒装句导致优质语料丢失，其他逻辑未动
                if 5 < abs(d - p) < 80:
                    pairs.append((p, d))
            
            unique_pairs = list(set(pairs))[:5] 
            for p, d in unique_pairs:
                tasks.append({
                    "type": "Variable Impact Analysis (Multi-hop)",
                    "context": self.ctx.get_multihop_context(p, d, 2)
                })

        # 2. [耦合诊断]
        coupled_cands = []
        for idx in range(len(self.structure)):
            item = self.structure[idx]
            if item.get('content_type') == 'invalid_chart': continue
            p_count = len([e for e in item.get('entities',[]) if e['category']=='PARAMETER'])
            if p_count >= 2: coupled_cands.append(idx)
        
        for idx in random.sample(coupled_cands, min(len(coupled_cands), 3)):
            tasks.append({
                "type": "Coupled Diagnosis",
                "context": self.ctx.get_window(idx, 2)
            })

        # 3. [图表/定量计算]
        for idx in random.sample(self.chart_indices, min(len(self.chart_indices), 4)):
            tasks.append({
                "type": "Quantitative Data Analysis",
                "context": self.ctx.get_window(idx, 2)
            })

        # 4. [工艺优化]
        if self.conclusion_indices:
            tasks.append({
                "type": "Process Optimization Strategy",
                "context": self.ctx.get_window(self.conclusion_indices[-1], 3)
            })

        # 5. [异常处理]
        if self.anomaly_indices:
            idx = random.choice(self.anomaly_indices)
            tasks.append({
                "type": "Anomaly Handling & Safety",
                "context": self.ctx.get_window(idx, 2)
            })

        # 6. [基础逻辑]
        mechs = self.idx_by_category.get('MECHANISM', [])
        for idx in random.sample(mechs, min(len(mechs), 3)):
            tasks.append({
                "type": "Basic Mechanism Logic",
                "context": self.ctx.get_window(idx, 2)
            })
            
        # 7. [横向对比]
        if len(self.chart_indices) >= 2:
            idx1, idx2 = random.sample(self.chart_indices, 2)
            tasks.append({
                "type": "Horizontal Comparison",
                "context": self.ctx.get_multihop_context(idx1, idx2, 1)
            })

        return tasks

def call_llm_api(task_data):
    """调用 DeepSeek 生成英文 QA"""
    dim_type = task_data['type']
    context = task_data['context']
    
    # 【修改点 2】：System Prompt 彻底转向连铸坯表面裂纹
    system_prompt = f"""
You are a senior expert researcher in continuous casting metallurgy, explicitly specializing in SLAB SURFACE CRACKS (e.g., longitudinal cracks, transverse corner cracks, spider cracks) and SURFACE QUALITY. 
Generate a high-quality Question-Answer pair based on the text.
Dimension: [{dim_type}]

Rules:
1. Output ENGLISH only.
2. Return JSON format.
3. The question must require reasoning (logical deduction) regarding surface defect mechanisms, mold parameters, or surface microstructure.
4. Include 'thought' (Chain of Thought).
5. Ensure the focus of the QA is entirely on surface or sub-surface phenomena.
6. STRICTLY PROHIBITED: Do NOT use phrases like "Based on the text", "According to the paper", "In the provided context", or "As mentioned in the excerpt" in your question or answer. The QA pair must read as a standalone, real-world metallurgical problem and expert answer.

Format:
{{
    "question": "...",
    "thought": "...",
    "answer": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nGenerate QA pair."}
            ],
            temperature=0.7,
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [API Error]: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    
    json_files = glob.glob(os.path.join(INPUT_DIR, '*.json'))
    print(f"Target: {MIN_QA_PER_PAPER}-{MAX_QA_PER_PAPER} QA pairs per paper.")
    print(f"Saving individual JSON files to: {OUTPUT_DIR}")
    print(f"Processing {len(json_files)} files...\n")
    
    total_files_processed = 0
    
    for i, file_path in enumerate(json_files):
        file_name = os.path.basename(file_path)
        print(f"[{i+1}/{len(json_files)}] Processing: {file_name} ...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            sampler = StrategySampler(paper_data)
            all_tasks = sampler.get_tasks()
            
            random.shuffle(all_tasks)
            selected_tasks = all_tasks[:MAX_QA_PER_PAPER]
            
            if len(selected_tasks) < MIN_QA_PER_PAPER:
                print(f"  - Warning: Only {len(selected_tasks)} logical points found.")
            
            print(f"  -> Generating {len(selected_tasks)} QAs...")
            
            paper_qa_list = []
            
            for task in selected_tasks:
                llm_result = call_llm_api(task)
                if llm_result:
                    # 【核心确认】：你的 CoT 思维链拼接逻辑完好无损地在这里！
                    entry = {
                        "instruction": llm_result['question'],
                        "input": f"Context:\n{task['context']}",
                        "output": f"{llm_result['thought']}\n\n{llm_result['answer']}",
                        "meta": {
                            "source_paper": file_name,
                            "dimension": task['type']
                        }
                    }
                    paper_qa_list.append(entry)
                    time.sleep(0.1) 

            if paper_qa_list:
                output_filename = f"QA_{file_name}"
                save_path = os.path.join(OUTPUT_DIR, output_filename)
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(paper_qa_list, f, ensure_ascii=False, indent=2)
                
                print(f"  -> Saved {len(paper_qa_list)} QAs to {output_filename}")
                total_files_processed += 1
            else:
                print("  -> No valid QAs generated for this file.")

        except Exception as e:
            print(f"  [Error] Processing file {file_name}: {e}")

    print(f"\n========================================")
    print(f"Completed! Processed {total_files_processed} files.")
    print(f"Check folder: {OUTPUT_DIR}")
    print(f"========================================")

if __name__ == "__main__":
    main()