import os
import json
import torch
import time
from datetime import datetime
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ================= 配置区域 =================
# ✅ 已修改为你提供的准确路径
# 使用 r"" 原始字符串格式，完美兼容 Windows 路径
MODEL_PATH = r"D:\models\InternVL2-4B"
BASE_OUT_DIR = r"D:\test_out"
# ==========================================

def build_transform(input_size=448):
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(list(target_ratios), key=lambda x: x[0] * x[1])
    best_ratio = (1, 1)
    min_dist = float('inf')
    for ratio in target_ratios:
        dist = abs(aspect_ratio - ratio[0] / ratio[1])
        if dist < min_dist:
            min_dist = dist
            best_ratio = ratio
        elif dist == min_dist:
            if ratio[0] * ratio[1] > best_ratio[0] * best_ratio[1]:
                best_ratio = ratio
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

class MetallurgyEnricher:
    def __init__(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 正在从 {MODEL_PATH} 加载 InternVL2-4B...")
        start_time = time.time()
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16, 
                trust_remote_code=True,
                device_map="cuda", 
                low_cpu_mem_usage=True
            ).eval()
        except OSError as e:
            print(f"\n❌ 路径错误! 请确认文件夹 '{MODEL_PATH}' 里面确实包含 config.json 文件。")
            print(f"错误详情: {e}")
            raise e

        self.transform = build_transform()
        self.dummy_img = Image.new('RGB', (448, 448), (255, 255, 255))
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 模型加载成功! (耗时: {time.time() - start_time:.2f}s)")

    def get_dynamic_prompt(self, content, block_type, caption):
        cap = caption.lower()
        if block_type == "IMAGE":
            if any(x in cap for x in ['crack', 'sem', 'microstructure', 'morphology', 'etched']):
                return (f"As a metallurgy expert, analyze this image ({caption}). "
                        "Identify: 1. Crack initiation sites (e.g., oscillation marks). "
                        "2. Propagation path (intergranular vs transgranular). "
                        "3. Signs of center segregation.")
            elif any(x in cap for x in ['curve', 'temperature', 'tbtr', 'cooling']):
                return (f"Analyze this chart ({caption}). "
                        "Focus on TBTR range (764-832 C) and cooling intensity.")
            return f"Describe this metallurgical figure: {caption}."
        elif block_type == "MATH":
            return f"Explain formula: {content}. Context: {caption}."
        elif block_type == "TABLE":
            return f"Summarize data: {content}. Context: {caption}."
        return f"Technical description for: {caption}."

    def process_folder(self, folder_path, current_idx, total_count):
        folder_name = os.path.basename(folder_path)
        print(f"\n{'='*50}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📄 进度 [{current_idx}/{total_count}] | 文献: {folder_name}")
        
        files = os.listdir(folder_path)
        md_name = next((f for f in files if f.startswith("refined_") and f.endswith(".md")), None)
        json_name = next((f for f in files if f.startswith("assets_") and f.endswith(".json")), None)
        
        if not (md_name and json_name):
            print("⚠️ 跳过: 文件不全")
            return

        md_path = os.path.join(folder_path, md_name)
        with open(os.path.join(folder_path, json_name), 'r', encoding='utf-8') as f:
            assets = json.load(f)
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        total = len(assets)
        doc_start = time.time()
        
        for idx, (placeholder, info) in enumerate(assets.items(), 1):
            t0 = time.time()
            print(f"   [{idx}/{total}] 分析 {placeholder}...", end="", flush=True)
            
            try:
                prompt = self.get_dynamic_prompt(info['raw_content'], info['type'], info['caption'])
                img_path = os.path.join(folder_path, info['raw_content']) if info['type'] == 'IMAGE' else None
                
                if img_path and os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    images = dynamic_preprocess(image, max_num=6)
                else:
                    images = [self.dummy_img.resize((448, 448))]
                
                pixel_values = [self.transform(img) for img in images]
                pixel_values = torch.stack(pixel_values).to(torch.float16).cuda()
                
                outputs = self.model.chat(
                    self.tokenizer, pixel_values, prompt, 
                    generation_config={'max_new_tokens': 512},
                    history=None, return_history=True
                )
                desc = outputs[0] if isinstance(outputs, tuple) else outputs

                if desc:
                    content = content.replace(placeholder, f"{placeholder}\n\n> **Expert Analysis:** {desc.strip()}\n")
                
                print(f" ✅ ({time.time() - t0:.2f}s)")
            except Exception as e:
                print(f" ❌ {e}")

        out_path = os.path.join(folder_path, md_name.replace("refined_", "enriched_"))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🎉 完成! 耗时: {time.time() - doc_start:.2f}s")
        torch.cuda.empty_cache()

    def run(self):
        subdirs = [os.path.join(BASE_OUT_DIR, d) for d in os.listdir(BASE_OUT_DIR) if os.path.isdir(os.path.join(BASE_OUT_DIR, d))]
        total = len(subdirs)
        print(f"📊 文献总数: {total}")
        for i, path in enumerate(subdirs, 1):
            self.process_folder(path, i, total)
        print("\n✨ 全部任务结束！")

if __name__ == "__main__":
    enricher = MetallurgyEnricher()
    enricher.run()