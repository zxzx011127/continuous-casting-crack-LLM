import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm

# ==========================================
# ⚙️ 配置区 (带上下文 / 开卷版)
# ==========================================
# 跑哪个模型，就改这里的路径
MODEL_PATH = r"D:\wxk_program\models\DeepSeek-R1-Distill-Qwen-7B" 
DATASET_PATH = "test_think.json"

# 🌟 每次测新模型，记得改一下输出文件名，防止覆盖
OUTPUT_PATH = "result_OpenBook_DeepSeek-R1-Distill-Qwen-7B.json" 

SYSTEM_PROMPT = "You are an expert in metallurgy and materials science. Please analyze the problem logically and provide your final answer strictly in English."
# ==========================================

def main():
    print(f"🚀 正在加载模型 (开卷测试): {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.eval()

    print(f"📂 正在读取测试集: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    results = []
    start_time = time.time()

    for idx, item in enumerate(tqdm(test_data, desc="🧠 开卷推理中")):
        
        instruction = item.get("instruction", "")
        context_input = item.get("input", "")
        ground_truth = item.get("output", "")
        
        # 开卷模式：拼装 问题 + 外部知识
        if context_input:
            prompt = f"{instruction}\n\n{context_input}"
        else:
            prompt = instruction

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=4096,  # 🌟 必须够大，给推导留足空间
                do_sample=True,       
                temperature=0.3,      # 🌟 0.3严谨模式，防幻觉
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id  # 🌟 消除红字警告，保护进度条
                # 🚫 注意：坚决不加 repetition_penalty，保护思维链！
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 🌟 原汁原味解码，保留模型生成的 <think> 标签，方便与清洗后的 ground_truth 对齐
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        results.append({
            "id": idx + 1,
            "instruction": instruction,
            "context_provided": "Yes",
            "model_prediction": response, # 这里包含完整的 <think>xxx</think>结论
            "ground_truth": ground_truth, # 注意：你的测试集这里的答案也需要手动套上 <think>
            "meta": item.get("meta", {})
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 开卷测试完成！耗时 {round(time.time() - start_time, 2)} 秒。已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()